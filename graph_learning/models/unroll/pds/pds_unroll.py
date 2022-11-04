"""
    PDS was introduced by  Kalofolias's "How to learn a graph from smooth signals", https://arxiv.org/abs/1601.02513

    It was unrolled by Pu in "Learning to Learn Graph Topologies". The implementation below borrow heavily from their
    released version, found at their github repo: https://github.com/xpuoxford/L2G-neurips2021/blob/master/src/models.py

"""

import math, sys, torch, pytorch_lightning as pl
from torch import nn
from pytorch_lightning import seed_everything
from typing import Optional, Any, Dict, List
from pathlib import Path

file = Path(__file__).resolve()
path2project = str(file.parents[4]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/

from graph_learning.misc.utils import num_edges2num_nodes
from graph_learning.models.model_based.smooth.pds import PDS as PDS_model_based
from graph_learning.models.unroll.gdn.gdn_utils import filter_repeats, construct_prior
from graph_learning.models.unroll.graph_unrolling_base import unrolling

#import torch_sparse


class pdsCell(nn.Module):
    def __init__(self,
                 use_vae: bool = False
                 ):
        super(pdsCell, self).__init__()
        self.use_vae = use_vae
        r = (torch.rand([3])-0.5)/10 # r ~ U[-.1, .1]
        alpha, beta, gamma = 1, 1, .01 # from paper
        self.init_param_vals = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        self.alpha_param = nn.Parameter(alpha + r[0], requires_grad=True)
        self.beta_param = nn.Parameter(beta + r[1], requires_grad=True)
        self.gamma_param = nn.Parameter(gamma + r[2]/1000, requires_grad=True)
        #nn.init.ones_(self.alpha)
        #nn.init.ones_(self.beta)
        self.output_zeros = False

    # y is the input data -> called z in o.g. l2g code
    # w is the input refined graph estimate
    # D is linear transform mapping vectorized adjacency into degree vector. Called S in LDPGs.
    def forward(self, y: torch.tensor, w: torch.tensor, v: torch.tensor, D: torch.tensor):
        w, v = PDS_model_based._shared_step_dong(y=y, w=w, v=v, D=D,
                                                 alpha=self.alpha_param,
                                                 beta=self.beta_param,
                                                 gamma=self.gamma_param,
                                                 eps=1e-8)
        self.output_zeros = False
        if torch.allclose(w, torch.zeros(1, device=w.device)) or \
                torch.allclose(v, torch.zeros(1, device=v.device)):
            print(f'outputting all zeros!')
            self.output_zeros = True
        return w, v

    def resample_params(self):
        r = (torch.rand([3]) - 0.5) / 10  # r ~ U[-.1, .1]
        alpha, beta, gamma = self.init_param_vals['alpha'], self.init_param_vals['beta'], self.init_param_vals['gamma']
        #alpha, beta, gamma = torch.tensor(1.0), torch.tensor(1.0), torch.tensor(.01)
        with torch.no_grad():
            print(f'\tResampling: alpha/beta/gamma {self.alpha:.3f}/{self.beta:.3f}/{self.gamma:.5f} -> {alpha:.3f}/{beta:.3f}/{gamma:.5f}')
            self.alpha.copy_(alpha + r[0])
            self.beta.copy_(beta + r[1])
            self.gamma.copy_(gamma + r[2]/1000)


class pds_unroll(unrolling, pl.LightningModule):
    """
    Paper: `Learning to Learn Graph Topologies
    <https://arxiv.org/abs/2110.09807>`_
    Paper authors: Xingyue Pu, Tianyue Cao, Xiaoyun Zhang, Xiaowen Dong, Siheng Chen
    Implemented by:
        - 'Max Wasserman <maxw14k@gmail.com>'_
    Args:
    """
    def __init__(self,
                 # architecture
                 depth: int,  # number of unrolled iterations
                 num_channels: int,
                 share_parameters: bool,
                 sparse: bool = False,
                 # loss, optimizer
                 loss: str = 'se',
                 monitor: str = 'val/se/mean',  # monitor is used by optimizer to check to reduce lr, stop running, etc
                 learning_rate: float = .03,
                 lr_decay: float = 0.95,
                 hinge_margin: float = .25, hinge_slope: float = 1.0,
                 intermed_loss_discount: float = 0.8,  # 'dn' in dong l2g code
                 l2_reg: float = 0.0,
                 # prior
                 n_train: Optional[int] = None,
                 prior: Optional[torch.Tensor] = None,
                 learn_prior: bool = None,
                 # reproducability
                 seed: int = 50,
                 # threshold finding
                 threshold_metric: str = 'acc',  # which metric to use when choosing threshold value
                 num_threshold_test_points: int = 200,  # how many points to check between min/max y_hat output
                 use_output_mlp: bool = False
                 ):
        super(unrolling, self).__init__()  # this calls unrolling init
        super(pl.LightningModule, self).__init__()
        self.unrolling_base_class = unrolling
        self.save_hyperparameters()

        if seed is not None:
            seed_everything(seed, workers=True)

        # construct layers in model
        layers = []
        assert num_channels == 1, f"MIMO PDS not yet implemented"
        print(f"\n-->depth {depth}, num_channels {num_channels} with {'' if share_parameters else 'OUT'} shared paramters.\n")
        if share_parameters:
            single_layer = pdsCell() # once mimo implemented feed in num_channels
            layers = [single_layer for _ in range(depth)]
        else:
            for layer in range(depth):
                # last layer no norm
                layers.append(pdsCell()) # c_in=1 if layer == 0 else num_channels

        self.module_list = nn.ModuleList(layers)

        if prior is not None:
            assert prior.ndim == 2, f'prior must be a 1D vector!'
            assert not learn_prior, f'can either take in prior or learn the prior, not both'

        if learn_prior:
            print(f'learning the prior! Think about initialization... Should all be positive??')
            num_edges = n_train * (n_train - 1) // 2
            self.training_prior = nn.Parameter(torch.randn(1, num_edges) / 10)
        else:
            self.register_buffer("training_prior", prior, persistent=True)
        self.checkpoint_loaded = False  # are we being loaded from a checkpoint?

        # best threshold found on training set when training_prior was constructed
        self.register_buffer('threshold', torch.tensor([-2.0]), persistent=True)

        # best threshold found on validation set after training done, to be used for testing
        # testing_threshold has to be attribute instead of registered buffer. The issue: when loading from load_from_
        # checkpoint, the value of the registered buffer did not update during the test_setp.
        self.testing_threshold = None
        self.min_output, self.max_output = torch.tensor([-1]), torch.tensor([1])

        # private variables to be used for printing/logging
        self.list_of_metrics = [] # on validation set

        if use_output_mlp:
            # f = # input features
            # h = hidden layer size
            f = num_channels + 2 # also include A_O and prior
            h = f
            self.mlp = nn.Sequential(nn.Linear(f, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh(),
                                     nn.Linear(h, 1))
        self.register_buffer("D", graph_learning.misc.utils.sumSquareForm(n_train), persistent=True)
        #self.Ds = {}
        #self.D = None

    def shared_step_mimo(self, batch, batch_idx, stage):
        x, y = batch[:2]
        batch_size, num_edges = x.shape[:-1]
        n = num_edges2num_nodes(num_edges)

        # prior is a vector. Initialize to zeros if none
        prior = self.training_prior if self.training_prior else torch.zeros(1, num_edges, device=self.device)
        w = prior.expand(batch_size, -1).unsqueeze(1)
        v = torch.zeros([batch_size, 1, n], dtype=w.dtype, device=device)

        c_in_dim = 2
        for i, layer in enumerate(self.module_list):
            #print(f'pds unroll: x {x.device} w {w.device} v {v.device}, input device {device}') # D {D.device()}')
            w, v = layer(y=x, w=w, v=v,  D=self.D)
            v, w = v.mean(dim=c_in_dim), w.mean(dim=c_in_dim)
        pred = w.mean(dim=1)
        return {'prediction': pred, 'intermediate_outputs': None}

    def shared_step(self, batch, batch_idx, stage):
        x, y = batch[:2]
        batch_size, num_edges = x.shape[:-1]
        n = num_edges2num_nodes(num_edges)

        # prior is a vector. Initialize to zeros if none
        prior = self.training_prior if self.training_prior else torch.zeros(1, num_edges, device=self.device)
        w = prior.expand(batch_size, -1)
        v = torch.zeros([1, n], dtype=w.dtype, device=device).expand(batch_size, -1)

        for i, layer in enumerate(self.module_list):
            # print(f'pds unroll: x {x.device} w {w.device} v {v.device}, input device {device}') # D {D.device()}')
            w, v = layer(y=x, w=w, v=v, D=self.D)
        pred = w
        return {'prediction': pred, 'intermediate_outputs': None}

    """
    
    def shared_step_old(self, batch, prior=None, device=None):
        if device is None:
            # when this LightningModule is used as a submodule, the self.device parameter is NOT appropriately updated,
            # thus optionally feed in device to handle this case. When used as submodule: pds(batch, device=self.device)
            device = self.device


        # v is intermediate ~degree estimate (or something like that in dual space)
        # w is intermediate graph estimate
        #x = adj2vec(batch[0])
        x = adj2vec(batch[0]) if batch[0].ndim == 3 else batch[0]

        assert x.ndim == 2, f'need x and y to be batch of vectorized adjs'
        batch_size, num_edges = x.shape
        n = num_edges2num_nodes(num_edges)
        D = self.find_transform(n, sparse=self.hparams.sparse)
        if torch.cuda.is_available():
            D = D.cuda()

        # v and w need to be consistent with each other. Set one, then run step or two of alg
        #  for consistent starting values. Add sampled/learned prior here.
        w = prior if prior != None else self.prior_prep(batch_size=batch_size, N=n)
        w = adj2vec(w) if w.ndim == 3 else w
        v = torch.zeros([batch_size, n], dtype=w.dtype, device=device).float()

        intermediate_outputs = torch.zeros(size=(self.hparams.depth, batch_size, num_edges), device=device)
        #print(f'\n\tdevice of w {w.device} v {v.device} D {D.device}')
        for i, layer in enumerate(self.module_list):
            #print(f'pds unroll: x {x.device} w {w.device} v {v.device}, input device {device}') # D {D.device()}')
            w, v = layer(y=x, w=w, v=v,  D=D)
            #intermediate_outputs.append(w)
            intermediate_outputs[i] = w

        return {'prediction': w, 'intermediate_outputs': intermediate_outputs}
    """
    def configure_optimizers(self):
        """
        lr_decay = self.hparams.lr_decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
        return [optimizer], {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def unrolling_trainable_parameters(self):

        params = []

        # this excludes parameters from any MLPs in the network
        for m in self.module_list:
            params.append(m.alpha_param.view(-1))
            params.append(m.beta_param.view(-1))
            params.append(m.gamma_param.view(-1))

        return torch.cat(params)

    def find_transform(self, n, sparse):
        if self.D is not None:
            #if type(self.D) == torch_sparse.tensor.SparseTensor:
            #    D_n = self.D.sizes()[0]
            #else:
            D_n = self.D.shape[0]
            if n == D_n:
                # check that it's the right size, otherwise construct another D to return
                return self.D

        # either its None or the wrong shape
        if self.D is None:
            self.D = graph_learning.misc.utils.sumSquareForm(N=n, out_sparse=sparse)

        if sparse:
            raise Exception("torch sparse not installed!")
            #self.D = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(self.D)

        return self.D

        """
        if n in self.Ds:
            return self.Ds[n]
        else:
            #D = graph_learning.misc.utils.sumSquareForm(n, out_sparse=True)
            D = graph_learning.misc.utils.sumSquareForm(N=n, out_sparse=sparse)  # l2g_utils.coo_to_sparseTensor(l2g_utils.get_degree_operator(m)).to(device)
        if sparse:
            D = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(D)
        self.Ds[n] = D

            return D
        """

    def construct_prior_(self, prior_dl=None, prior_dtype=torch.float32):
        # constructing prior from data
        if self.hparams.prior_construction in ['mean', 'median', 'multi']:
            # create prior with prior_ds set (train set): prior_dl cannot be none
            assert prior_dl is not None, f'to construct prior from data, need train data'
            _, prior_scs, prior_subject_ids, _, _ = prior_dl.dataset.full_ds()
            unique_scs_train_set = filter_repeats(prior_scs, prior_subject_ids)
            if self.hparams.prior_construction in ['mean', 'median']:
                prior = construct_prior(unique_scs_train_set, frac_contains=0, reduction=self.hparams.prior_construction)
                self.training_prior[0] = prior[0]
                prior_channels = [prior]
                prior_channel_names = [self.hparams.prior_construction]
            else:  # multi
                self.training_prior[0, 0] = construct_prior(unique_scs_train_set, frac_contains=self.hparams.prior_frac_contains, reduction='mean')
                self.training_prior[1, 0] = construct_prior(unique_scs_train_set, frac_contains=self.hparams.prior_frac_contains, reduction='median')
                prior_channels = [self.training_prior[0, 0], self.training_prior[1, 0]]
                prior_channel_names = ['mean', 'median']
        # these priors do not use real data. Reconstructed each time, only need graph size N.
        # This is useful when we want to test on different sized data than we trained on: simply feed in test set as
        # val_dl, and prior will be constructed appropriately.
        else:
            if self.hparams.prior_construction == 'block':
                block_scale = .35  # minimizes se
                assert self.hparams.n_train % 2 == 0, f'for block prior, n must be even (or in general divisible by number of communities'
                ones = torch.ones((self.hparams.n_train // 2), (self.hparams.n_train // 2))
                # self.prior = torch.block_diag(ones, ones).view(1, N, N)*block_scale
                prior_channels = [torch.block_diag(ones, ones).view(1, self.hparams.n_train, self.hparams.n_train) * block_scale]
                prior_channel_names = ['block']
                # for sc in np.arange(0.3, .4, .01):
                #    print(f'scale: {sc}', predicition_metrics(y_hat=sc*torch.block_diag(ones, ones).view(1, N, N).repeat(len(val_scs), 1, 1), y=val_scs, y_subject_ids=val_subject_ids))
            if self.hparams.prior_construction == 'sbm':
                prob_matrix = prior_dl.dataset.prob_matrix()
                prior_channels = [prob_matrix.expand(1, self.hparams.n_train, self.hparams.n_train).to(prior_dtype)]
                prior_channel_names = ['sbm']
            elif self.hparams.prior_construction in ['zeros']:
                # self.prior = torch.zeros(1, N, N, dtype=train_fcs.dtype)
                prior_channels = [torch.zeros(1, self.hparams.n_train, self.hparams.n_train, dtype=prior_dtype)]
                prior_channel_names = ['zeros']
            elif self.hparams.prior_construction in ['ones']:
                prior_channels = [torch.ones(1, self.hparams.n_train, self.hparams.n_train, dtype=prior_dtype)]
                prior_channel_names = ['ones']
            else:
                raise ValueError('unrecognized prior construction arg')
            self.training_prior[0] = prior_channels[0]

        return prior_channels, prior_channel_names

    def prior_prep(self, batch_size, N):
        # called by train_step/val_step/test_step to construct the prior tensor.

        # By NOT using self.training_prior (constructed in setup() during initial training) when we don't
        # need to (e.g. for mean/median), then we can test on graphs of sizes different than those trained on.
        prior_channels = 1 #if (not self.hparams.share_parameters) else self.layers[0].c_in
        if self.hparams.prior_construction == 'zeros':
            return torch.zeros(size=(batch_size, N, N), device=self.device).expand(batch_size, N, N) #prior_channels, batch_size, N, N)
        elif self.hparams.prior_construction == 'ones':
            return 0.5 * torch.ones(size=(batch_size, N, N), device=self.device).expand(batch_size, N, N) #prior_channels, batch_size, N, N)
        elif self.hparams.prior_construction == 'block':
            block_scale = .35  # minimizes se for brain graphs
            assert (N % 2) == 0, f'block diagram requires even N'
            ones = torch.ones(size=(batch_size, (N // 2), (N // 2)), device=self.device)
            return torch.block_diag(ones, ones).expand(prior_channels, batch_size, N, N) * block_scale
        elif self.hparams.prior_construction in ['mean', 'median', 'sbm']:
            # must use training_prior
            assert self.training_prior.shape[-1] == N, f'when using prior constructed from data, we cannot test on data of different size than it'
            return self.training_prior.expand(batch_size, N, N) # (prior_channels, batch_size, N, N)


#### unrolling module from l2g authors ####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class l2g_unrolling_author_code(nn.Module):
    """
        implimentation from paper
    """

    def __init__(self, num_unroll):
        super(l2g_unrolling_author_code, self).__init__()

        self.layers = num_unroll  # int

        self.gn = nn.Parameter(torch.ones(num_unroll, 1)/100, requires_grad=True)

        self.beta = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.beta)

        self.alpha = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.alpha)

    def prox_log_barrier(self, y, gn, alpha):
        up = y ** 2 + 4 * gn * alpha
        up = torch.clamp(up, min=1e-08)
        return (y - torch.sqrt(up)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def forward(self, z):
        batch_size, l = z.size()

        z = pds_unroll.check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = pds_unroll.coo_to_sparseTensor(pds_unroll.get_degree_operator(m)).to(device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(device)
        w_list = torch.empty(size=(batch_size, self.layers, l)).to(device)

        for i in range(self.layers):

            y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
            y2 = v + self.gn[i] * torch.matmul(w, D.T)

            p1 = torch.max(zero_vec, y1)
            #assert torch.allclose(p1, F.relu(y1))
            p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

            q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
            q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        return w_list


if __name__ == "__main__":
    print('l2g main loop')

