""""
    GDN is an unrolling of proximal gradient iterations introduced by Wasserman et al in
    "Learning Graph Structure from Convolutional Mixtures", https://arxiv.org/abs/2205.09575.

    It posits a polynomial relation between the observed graph and the latent graph. Thus far it is the *only* unrolling
    which is a neural network, i.e. a linear transform followed by pointwise non-linearity. All other unrolling involve 
    non-linear transformations in their layers.
    
    GDN is also unique in that it unrolls a non-convex objective. Thus there is no model-based GDN formulation that can 
    be easily tested. 

"""

from typing import List, Any, Optional
import math, sys, numpy as np, torch, pytorch_lightning as pl
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import seed_everything
from typing import Any, Dict, List, Optional, Type

"""
from pathlib import Path
file = Path(__file__).resolve()
path2project = str(file.parents[4]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/
"""

from graph_learning.misc.utils import adj2vec, vec2adj, normalize_slices, mimo_tensor_polynomial, adjs2fvs
from graph_learning.misc.metrics import best_threshold_by_metric, hinge_loss

from graph_learning.misc.metrics import compute_metrics
from graph_learning.models.unroll.graph_unrolling_base import unrolling


def clamp_tau(model, large_tau):
    for name, W in model.named_parameters():
        if 'tau' in name:
                W.clamp_(min=0, max=large_tau)


class gdnCell(nn.Module):

    def __init__(self,
                 c_in: int = 1,
                 c_out: int = 1,
                 order_poly_fc: int = 1,
                 learn_tau: bool = True):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out

        # parameters
        self.stdv_scaling = 1 / 3  # scale N(0,1)*stdv_sampling-> N(0,stdv_sampling^2)
        self.init_alpha_sample_mean, self.init_beta_sample_mean = 1, 0

        # alpha scales the input recieved from last layer. Make it ~1 -> making small changes to previous layer.
        self.alpha = nn.Parameter(torch.randn(c_out, c_in) * self.stdv_scaling + self.init_alpha_sample_mean)  # N(1, 1/9)
        # beta scales the part of the gradient. Make it ~0.
        self.beta = nn.Parameter(torch.randn(c_out, c_in) * self.stdv_scaling + self.init_beta_sample_mean)

        # these define the coefficients of the polynomial of the FC. Make these all ~0 except for the 1st order term,
        # which should be ~1. because we group it with an unscaled A_in. See paper.
        # Note that 0th order term is NOT used because it only affects diagonal, but will be included in parameter
        # count by pytorch.
        # coeffs_poly_fc[i] in R^(order+1 x c_in) are all polynomial coeffs used by output channel i.
        # coeffs_poly_fc[i, j] in R^c_in are the poly coeffs corresponding to jth polynomial basis
        # coeffs_poly_fc[i, j, k] in R is the poly coeff corresponding to jth polynomial basis for the
        # k^th input channel
        self.coeffs_poly_fc = nn.Parameter(torch.randn(c_out, order_poly_fc + 1, c_in) * self.stdv_scaling)
        with torch.no_grad():
            self.coeffs_poly_fc[:, 1, :] = 1

        self.learn_tau = learn_tau
        tau = torch.linspace(start=.01, end=.9, steps=c_out).view(c_out, 1, 1, 1)
        if learn_tau:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32), persistent=True)

        self.output_zeros = False

    def forward(self, s_in, a_o, normalize=True):
        # s_in = [c_in, batch_size, N, N]
        # a_o = [batch_size, N, N]
        assert (len(s_in.shape) == 4) and (len(a_o.shape) == 3) and (s_in.shape[1] == a_o.shape[0])
        assert (s_in.shape[-1] == s_in.shape[-2]) and (a_o.shape[-1] == a_o.shape[-2])
        c_in = s_in.shape[0]
        assert (c_in == self.c_in), f'input has {c_in} input channels but expected {self.c_in} input channels'
        batch_size, N, _ = a_o.shape
        assert torch.all(self.tau >= 0), f'tau is negative {self.tau}'

        # c0*I + c1*A_O + c2*A_O^2 + ...
        poly_a_o = mimo_tensor_polynomial(a_o.expand(s_in.shape), self.coeffs_poly_fc, cob='cheb') #self.poly_basis)

        # shape of intermediate tensors: [c_out, c_in, batch_size, N, N]
        intermed_shape = (self.c_out, self.c_in, batch_size, N, N)
        a, a_o = s_in.expand(intermed_shape), a_o.expand(intermed_shape)

        temp = \
            self.alpha.view(self.c_out, self.c_in, 1, 1, 1) * a \
            + self.beta.view(self.c_out, self.c_in, 1, 1, 1) * (a.matmul(a_o) + a_o.matmul(a)) \
            + poly_a_o

        # Project onto adjacencies with no zero'd diagonal
        # zd = 0 on on slice diagonals, 1s everywhere else.
        zd = (torch.ones((N, N), device=s_in.device) - torch.eye(N, device=s_in.device)).expand(intermed_shape)
        temp = temp * zd

        # this can be a mean or sum reduction over the channels
        temp = torch.mean(temp, dim=1)
        if normalize:
            temp = normalize_slices(temp, which_norm='max_abs')
        s_out = F.relu(temp - self.tau)

        self.output_zeros = False
        if torch.allclose(s_out, torch.zeros(s_out.shape, device=s_out.device)):  # and False:
            self.output_zeros = True
            print('==================================')
            print('WARNING: A layer is outputting an all 0 S_out')
            # print(cdp(self.tau), 'tau: threshold cutoff')
            print('==================================')

        return s_out

    def resample_params(self, stdv_scaling=1 / 3, alpha_mean=1, beta_mean=0):
        module = self
        for name, param in module.named_parameters():
            with torch.no_grad():
                if ('tau' in name) and self.learn_tau:
                    param.copy_(param * 0.1)  # reduce it for less sparsity
                if 'alpha' in name:
                    new_param = module.stdv_scaling * torch.randn_like(param) + module.init_alpha_sample_mean
                    param.copy_(new_param)
                if 'beta' in name:
                    new_param = module.stdv_scaling * torch.randn_like(param) + module.init_beta_sample_mean
                    param.copy_(new_param)


class gdn(unrolling, pl.LightningModule):
    def __init__(self,
                 # architecture
                 depth: int,
                 num_channels: int,
                 share_parameters: bool,
                 poly_fc_order: int = 1,
                 learn_tau: bool = True,
                 # loss, optimizer
                 loss: str = 'hinge',
                 monitor: str = 'error',  # monitor is used by optimizer to check to reduce lr, stop running, etc
                 learning_rate: float = .01,
                 hinge_margin: float = 1.0, hinge_slope: float = 1.0,
                 l2_reg: float = 0.0,
                 # subnetwork of interest: None for synthetics.
                 # prior
                 n_train: int = 68,
                 prior: Optional[torch.Tensor] = None, # Assume prior is given beforehand if to be used
                 learn_prior: bool = None,
                 # reproducability
                 seed: int = 50,
                 # threshold finding
                 threshold_metric: str = 'acc',  # which metric to use when choosing threshold value
                 num_threshold_test_points: int = 200, # how many points to check between min/max y_hat output
                 use_output_mlp: bool = False
                 ):
        super(unrolling, self).__init__() #this calls unrolling init
        super(pl.LightningModule, self).__init__()
        self.unrolling_base_class = unrolling
        self.save_hyperparameters()
        seed_everything(seed, workers=True)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1225
        # for logging of hparams and computationa graph
        # self.example_input_array = torch.zeros(10, 68, 68)

        if seed is not None:
            seed_everything(seed, workers=True)
        
        layers = []
        print(f"\n-->depth {depth}, num_channels {num_channels} with {'' if share_parameters else 'OUT'} shared paramters.\n")
        if share_parameters:
            single_layer = gdnCell(c_in=num_channels, c_out=num_channels, order_poly_fc=poly_fc_order, learn_tau=learn_tau)
            layers = [single_layer for _ in range(depth)]
        else:
            for layer in range(depth):
                # last layer no norm
                layers.append(gdnCell(c_in=1 if layer == 0 else num_channels, c_out=num_channels,
                                      order_poly_fc=poly_fc_order, learn_tau=learn_tau))

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

        # best threshold found on training set when training_prior was constructed
        self.register_buffer('threshold', torch.tensor([-2.0]), persistent=True)

        # best threshold found on validation set after training done, to be used for testing
        # testing_threshold has to be attribute instead of registered buffer. The issue: when loading from load_from_
        # checkpoint, the value of the registered buffer did not update during the test_setp.
        self.testing_threshold = None
        self.min_output, self.max_output = torch.tensor([-1]), torch.tensor([1])

        # private variables to be used for printing/logging
        self.list_of_metrics = []  # on validation set

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

        self.checkpoint_loaded = False  # are we being loaded from a checkpoint?

    #def setup(self, stage: Optional[str]):

    def on_train_start(self) -> None:
        y_train = self.trainer.datamodule.train_dataloader().dataset.adj.to(self.device)
        y_val = self.trainer.datamodule.val_dataloader().dataset.adj.to(self.device)
        y_train, y_val = adj2vec(y_train), adj2vec(y_val)
        # costruct tensor of labels from training and validation set
        means, stdes = self.prior_performance(prior=self.training_prior,
                                              y_train=y_train,
                                              y_val=y_val)
        for metric_name in means:
            self.log(name='prior/' + metric_name + '/mean', value=means[metric_name])
            self.log(name='prior/' + metric_name + '/mean', value=stdes[metric_name])

        if self.hparams.learn_prior:
            with torch.no_grad():
                # STILL NEED THIS??
                # initilized with some noise to break symmetry problem. Add on top of this.
                self.learned_prior.data += adj2vec(self.prior_channels[0]).to(self.learned_prior.device)
                # display
                # import matplotlib.pyplot as plt
                # plt.imshow(graph_learning_utils.vec2adj(self.learned_prior.data, self.hparams.n_train).squeeze())

    def shared_step(self, batch, int_out=False, apply_sigmoid=False):
        x, y = batch[:2]
        batch_size, N = x.shape[:-1]
        # prior is a vector. Initialize to zeros if none
        prior = vec2adj(self.training_prior, N).expand(batch_size, -1, -1).unsqueeze(0)
        s_out = prior
        for i, layer in enumerate(self.module_list):
            # dont normalize last layer
            s_out = layer(s_out, x, normalize = (i < (self.hparams.depth - 1)))

        pred = s_out.mean(dim=0)
        return {'prediction': pred, 'intermediate_outputs': None}

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: Optional[int] = 0) -> None:

        # over-ride base class - so have to manually call it
        self.unrolling_base_class.on_train_batch_end(self, outputs, batch, batch_idx, unused)

        # ensure tau stays in [0. ~.99]
        if self.hparams.learn_tau:
            clamp_tau(self, large_tau=.99)

        if self.hparams.learn_prior and False:
            # inspect gradients
            import matplotlib.pyplot as plt
            plt.imshow(vec2adj(self.learned_prior.grad, self.hparams.n_train).squeeze())
            plt.colorbar()

        return

    def configure_optimizers(self):
        #print(f'test with adam betas at default')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(.85, .99))  # , weight_decay=self.weight_decay)
        return optimizer

    def unrolling_trainable_parameters(self):

        params = []

        # this excludes parameters from any MLPs in the network
        for m in self.module_list:
            params.append(m.alpha.detach().view(-1))
            params.append(m.beta.view(-1))
            if m.learn_tau:
                params.append(m.tau.view(-1))

        return torch.cat(params)
    def log_parameter_distrib(self):
        """
            pull out all the parameters from the model and logs their histogram and summary statistics
        """
        import wandb

        alpha_param, beta_param, tau_param = [], [], []
        for m in self.module_list:
            alpha_param.append(m.alpha.detach().cpu().numpy().flatten())
            beta_param.append(m.beta.detach().cpu().numpy().flatten())
            tau_param.append(m.tau.detach().cpu().numpy().flatten())

        cat = np.concatenate
        alpha_param, beta_param, tau_param = cat(alpha_param), cat(beta_param), cat(tau_param)

        #exp = np.exp
        #alphas, betas = exp(alpha_param), exp(beta_param)
        #ratios = alphas / betas

        wandb.log({"alpha": wandb.Histogram(alpha_param)})
        wandb.log({"beta": wandb.Histogram(beta_param)})
        wandb.log({"tau": wandb.Histogram(tau_param)})
        #wandb.log({"exp(alpha)": wandb.Histogram(alphas)})
        #wandb.log({"exp(beta)": wandb.Histogram(betas)})
        #wandb.log({"ratios": wandb.Histogram(ratios)})

        def create_stats_dict(param_name, a):
            return {f'param/{param_name}/min': a.min(),
                    f'param/{param_name}/max': a.max(),
                    f'param/{param_name}/mean': a.mean(),
                    f'param/{param_name}/median': np.median(a),
                    f'param/{param_name}/stdev': np.std(a)}

        self.log_dict(create_stats_dict('alpha', alpha_param))
        self.log_dict(create_stats_dict('beta', beta_param))
        self.log_dict(create_stats_dict('tau', tau_param))

    """
    def on_after_backward(self) -> None:
        if self.hparams.use_output_mlp:
            max_grad = 0.0
            for i in range(len(self.mlp)):
                if 'Linear' in str(self.mlp[i]):
                    if self.mlp[i].weight.grad is not None:
                        max_grad = max(max_grad, self.mlp[i].weight.grad.abs().max().item())
            print(f'MLP max magn gradient: {max_grad:.6f}')

        max_grad = 0.0
        for layer in self.module_list:
            for name, params in layer.named_parameters():
                if 'alpha' in name or 'beta' in name or 'tau' in name or 'fc' in name:
                    max_grad = max(max_grad, params.grad.abs().max().item())

        #print(f'GDN max magn gradient: {max_grad:.6f}')
    """

    def prior_performance(self, prior, y_train, y_val, num_threshold_discretizations=300, threshold_metric='acc'):
        """
            Given the prior, find how well it does on the validation set
            For link-pred, first find threshold with training set
        """
        # how well does the prior do on th set?
        test_points = torch.linspace(start=prior.min().item(), end=prior.max().item(),
                                     steps=num_threshold_discretizations)

        threshold = best_threshold_by_metric(
            y_hat=prior.expand(y_train.shape[0], -1), y=y_train,
            thresholds=test_points,
            metric=threshold_metric,
            device=prior.device
        )

        metrics = compute_metrics(y_hat=prior.expand(y_val.shape[0], -1), y=y_val + 0.0, threshold=threshold,
                                  self_loops=False)
        means, stdes = {}, {}
        for metric_name, metric_values in metrics.items():
            means[metric_name] = metric_values.mean()
            stdes[metric_name] = torch.std(metric_values) / math.sqrt(len(metric_values))

        print(f"Prior performance on validation set: se {means['se']:.3f}, se/per_edge {means['se_per_edge']:.3f}, nse {means['nse']:.3f}, error {means['error']:.3f}")
        return means, stdes


if __name__ == "__main__":
    print('gdn main loop')

