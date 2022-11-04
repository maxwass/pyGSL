"""
    GLAD was the first unrolling for Graph Structure Learning to appear in the literature. It was introduced by
    Shrivastava et al in "GLAD: Learning Sparse Graph Recover", https://arxiv.org/abs/1906.00271.

    GLAD unrolls AM iterations on the GLASSO objective. It requires Positive Definiteness of intermedate graph outputs.
    It uses matrix square root operation. Exact matrix square root operations are expensive - in both the forward and
    backward pass - and we provide fast approximate alternatives for this which do not seem to degrade performance. See
    matrix_sqrt.py in this folder for more info.


    TODO: GLAD still must be intergrated into the graph_unrolling_base. Bit trickier because outputs are now
    positve definite matrices, with (i) possibly negative entries (ii) non-zero diagonal. Other methods assume symmetric
    adjacency with non-negative entries and zero-diagonal (no self loops). Metrics for link-prediction become more
    difficult to define - as classifying an edge prediction can be based on magnitude or signed magnitude - and
    regression metrics must be updated to account for diagonal.
"""

import math, sys, numpy as np, torch, pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import seed_everything
from typing import Optional, Any, Dict, List
from pathlib import Path
file = Path(__file__).resolve()
path2project = str(file.parents[4]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/

from graph_learning.misc.metrics import best_threshold_by_metric, compute_metrics, hinge_loss, regression_metrics, symmetric_classification_metrics
from graph_learning.models.unroll.glad.matrix_sqrt import MatrixSquareRoot
from graph_learning.misc.utils import adj2vec, vec2adj, adjs2fvs
sqrtm = MatrixSquareRoot.apply
try:
    from graph_learning.models.unroll.gdn.gdn_utils import filter_repeats, format_color, construct_prior, apply_mask, \
        print_subnet_perf_dict, percent_change_metrics
except:
    from graph_learning.models.unroll.gdn.gdn_utils import filter_repeats, format_color, construct_prior, apply_mask, \
        print_subnet_perf_dict, percent_change_metrics


def shallowest_layer_all_zero(model):
    #starting from beginning of model, check if the layer output all zeros.
    #  Return layer depth or -1
    for i, module in enumerate(model.module_list):
        if module.output_zeros:
            print(f"\t\t{i}th layer!")
            return i
    return -1

def rhoNN(f: int, h: int, dtype):
    # f = # input features
    # h = hidden layer size
    return nn.Sequential(nn.Linear(f, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, 1, dtype=dtype), nn.Sigmoid())


def lambdaNN(f: int, h:int, dtype):
    # f = # input features
    # h = hidden layer size
    return nn.Sequential(nn.Linear(f, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, 1, dtype=dtype), nn.Sigmoid())


class gladCell(torch.nn.Module):
    def __init__(self, h: int,  # hidden layer size
                       dtype: torch.dtype
                 ):
        super(gladCell, self).__init__()
        self.lambda_nn, self.rho_nn = lambdaNN(f=2, h=h, dtype=dtype), rhoNN(f=3, h=h, dtype=dtype)
        self.output_zeros = False

    def forward(self, sigma: torch.tensor, theta: torch.tensor, z: torch.tensor, lambda_: torch.tensor, non_neg: bool):
        batch_size, n = sigma.shape[:2]
        # || Z - Theta ||_F^2
        d = torch.sum((z-theta)**2, dim=(1, 2)).unsqueeze(dim=-1)
        # properly structure  d & l to be fed into lambda_nn in batched manner
        #  d.view(), l.view()
        assert d.shape == lambda_.shape == (batch_size, 1)
        lambda_ = self.lambda_nn(torch.cat([d, lambda_], dim=1))
        y = (1/lambda_).view(batch_size, 1, 1) * sigma - z

        I = torch.eye(n, device=sigma.device).expand(batch_size, n, n)
        theta = 0.5*(- y + sqrtm(torch.bmm(y.transpose(-1, -2), y) + (4/lambda_).view(batch_size, 1, 1) * I))

        # construct feature vectors by vectorizing and concat theta, sigma, & z
        fvs = adjs2fvs([theta, sigma, z]) # (bs * n * n, 3)
        rho = self.rho_nn(fvs).view(batch_size, n, n) # (bs * n * n, 1) -> (bs, n, n)
        z = torch.sign(theta) * torch.relu(torch.abs(theta) - rho)
        z = torch.relu(z) if non_neg else z

        # check if we are stuck on an all zeros output
        self.output_zeros = False
        if torch.allclose(z, torch.zeros(1, device=z.device)) or \
                torch.allclose(theta, torch.zeros(1, device=theta.device)):
            self.output_zeros = True

        return theta, z, lambda_


class glad(pl.LightningModule):
    def __init__(self,
                 # architecture
                 depth: int,  # number of unrolled iterations
                 h: int,  # hidden layer size
                 theta_init_offset: float, # add this to sample_cov to ensure invertible for init of theta_0
                 share_parameters: bool,
                 non_neg_outputs: bool = False, # enforce non-negativity of all (including intermed) outputs?
                 lambda_init: float = 1.0,
                 which_normalization: Optional[str] = None,
                 norm_last_layer: bool = False,
                 prior_construction: Optional[str] = None, #'mean',
                 n_train: Optional[int] = None,
                 # loss, optimizer
                 loss: str = 'hinge',
                 monitor: str = 'val/se/mean',  # monitor is used by optimizer to check to reduce lr, stop running, etc
                 optimizer: str = 'adam',
                 learning_rate: float = .03,
                 adam_beta_1: float = .9, adam_beta_2: float = .999, momentum: float = .95,
                 hinge_margin: float = .25, hinge_slope: float = 1.0,
                 gamma: float = 0.8,
                 # metrics
                 report_off_diag_metrics: bool = True,
                 # reproducability
                 seed: int = 50,
                 # threshold
                 threshold_metric: str = 'f1' #which metric to use when choosing threshold value
                 ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None:
            seed_everything(seed, workers=True)

        ### prior
        assert prior_construction in [None, 'mean']
        #if prior_construction is not None:
        #    assert n_train is not None, f'if using prior, must provide size of prior'
        #    self.register_buffer('training_prior', -1 * torch.ones(size=(1, n_train, n_train)), persistent=True)
        #    self.register_buffer('training_prior', -1 * torch.ones(size=(1, n_train, n_train)),
        #                         persistent=True)
        num_prior_channels = 1
        self.register_buffer('training_prior', -1 * torch.ones(size=(num_prior_channels, n_train, n_train)),
                             persistent=True)
        # best threshold found on training set when training_prior was constructed
        self.register_buffer('training_prior_threshold', -1 * torch.ones(num_prior_channels, 1), persistent=True)
        # best threshold found on validation set after training done, to be used for testing
        self.register_buffer('testing_prior_threshold', -1 * torch.ones(num_prior_channels, 1), persistent=True)

        self.prior_channels, self.prior_channel_names = None, None

        # self.register_buffer('real_data_prior', torch.zeros(1, 68, 68), persistent=True)
        self.register_buffer('threshold', torch.tensor([-1.0]), persistent=True)
        self.test_threshold = None  # placeholder, will save in checkpoint and load when needed
        # self.register_buffer('test_threshold', torch.tensor([-1.0]), persistent=True)
        ######
        self.checkpoint_loaded = False  # are we being loaded from a checkpoint?



        self.theta_init_offset = nn.Parameter(torch.Tensor([theta_init_offset]))

        # best threshold found on training set when training_prior was constructed
        self.register_buffer('threshold', torch.tensor([-2.0]), persistent=True)

        # best threshold found on validation set after training done, to be used for testing
        # testing_threshold has to be attribute instead of registered buffer. The issue: when loading from load_from_
        # checkpoint, the value of the registered buffer did not update during the test_setp.
        #self.register_buffer('testing_threshold', -2*torch.tensor(1), persistent=True)
        #self.register_buffer('testing_threshold', torch.tensor([-2.0]), persistent=True)
        self.testing_threshold = None
        self.min_output = torch.tensor([-1])
        self.max_output = torch.tensor([1])

        # private variables to be used for printing/logging
        self.list_of_metrics = [] # on validation set

        # SHOULD WE MAKE GLAD LAYER WITH THE 2 NN INSIDE?
        layers = []
        if share_parameters:
            #self.channels[0] = 1
            single_layer = gladCell(h=h, dtype=self.dtype)
            for layer in range(depth):
                layers.append(single_layer)
        else:
            for layer in range(depth):
                # last layer no norm
                layers.append(gladCell(h=h, dtype=self.dtype))

        self.module_list = nn.ModuleList(layers)

        self.learning_rate_reduce = {'reduce': False, 'count': 0}

    def setup(self, stage: Optional[str] = None) -> None:
        # dont use prior_channels/names yet
        #if stage == 'fit':
        #    if self.hparams.prior_construction is not None:
        #        prior_channels, prior_channel_names = self.construct_prior_(prior_dl=self.trainer.datamodule.train_dataloader())

        if stage in ['fit'] and not self.checkpoint_loaded:
            # assert torch.allclose(self.training_prior, -1*torch.ones(1)), f"init'ed to all -1. This is how we know not used yet."
            self.prior_channels, self.prior_channel_names = self.construct_prior_(prior_dl=self.trainer.datamodule.train_dataloader())
            try:
                train_scs, val_scs = self.trainer.datamodule.train_dataloader().dataset.full_ds()[1], \
                                     self.trainer.datamodule.val_dataloader().dataset.full_ds()[1]
            except AttributeError:  # AttributeError: 'Subset' object has no attribute 'full_ds'
                train_scs = torch.cat([a[1].unsqueeze(dim=0) for a in self.trainer.datamodule.train_dataloader().dataset], dim=0)
                val_scs = torch.cat([a[1].unsqueeze(dim=0) for a in self.trainer.datamodule.val_dataloader().dataset], dim=0)

            """
            for i, (prior_channel, prior_channel_name) in enumerate(zip(self.prior_channels, self.prior_channel_names)):
                # find performance of prior channel on validation set by finding optimal threshold (on training set) and
                # reporting performance
                self.training_prior_threshold[i], self.prior_metrics_val[prior_channel_name] = \
                    self.prior_performance(prior_channel=prior_channel, prior_channel_name=prior_channel_name,
                                           holdout_set_threshold=train_scs, holdout_set_metrics=val_scs)
            """
    def forward(self, batch) -> Any:
        return self.shared_step(batch=batch)

    def shared_step(self, batch, int_out=False):
        # theta_pred <-> z
        # Sb <-> theta
        # b  <-> t
        x, y = batch[:2]

        assert x.ndim == 3, f'need x and y to be batch of adjs'
        batch_size, n = x.shape[:2]
        # a single lambda is used for each sample
        lambda_ = torch.ones(batch_size, 1, device=self.device) * self.hparams.lambda_init
        I = torch.eye(n, device=self.device).expand(batch_size, n, n)
        if self.hparams.prior_construction is not None:
            theta = self.prior_prep(batch_size=batch_size, N=n) + self.theta_init_offset*I
        else:
            theta = torch.inverse(x + self.theta_init_offset*I)
        z = theta.detach().clone()
        intermediate_outputs = []
        for i, layer in enumerate(self.module_list):
            theta, z, lambda_ = layer(sigma=x, theta=theta, z=z, lambda_=lambda_, non_neg=self.hparams.non_neg_outputs)# self.trainer.datamodule.non_neg_labels)
            intermediate_outputs.append(z)

        return (z, intermediate_outputs) if int_out else z

    def intermediate_outputs(self, batch):
        return self.shared_step(batch, int_out=True)[1]

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]

        y_hat, intermediate_outputs = self.shared_step(batch, int_out=True)
        y_hat_vec = adj2vec(y_hat.detach())
        self.max_output = y_hat_vec.max() if not torch.isnan(y_hat.max()) else torch.tensor([100]) # IGNORE DIAG IN MAX SEARCH
        self.min_output = y_hat_vec.min() if not torch.isnan(y_hat.min()) else torch.tensor([-100])
        loss = self.compute_loss(intermediate_outputs=intermediate_outputs, y=y, use_raw_adj=True, per_sample_loss=True)
        self.threshold = self.threshold.to(self.device)
        metrics = self.compile_metrics(y_hat=y_hat.detach(), y=y, threshold=self.threshold)

        return {'loss': loss, 'metrics': metrics, 'batch_size': len(x)}

    def on_after_backward(self) -> None:
        """
        # log parameter values and gradients
        for i, m in enumerate(self.module_list):
            # m is a module ('layer') in the module list ('unrolling')
            for name, param in m.named_parameters():
                if param.requires_grad:
                    name = name + ('' if self.hparams.share_parameters else f'_{i}')
                    self.log(name='param/value/' + name, value=param.data)
                    if param.grad is not None: # None for last layer bc not used
                        self.log(name='param/grad/' + name, value=param.grad)

        for name, grad in self.named_parameters():
            if param.requires_grad:
                self.log(name='param/value/' + name, value=param.data)
            if param.grad is not None:
                self.log(name='param/grad/' + name, value=param.grad)

        """

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        # find the shallowest layer where model outputting ALL zeros. -1 if no layer outputting all zeros.
        sl = shallowest_layer_all_zero(self)

        if sl > -1:
            #sl = shallowest_layer_all_zero(self)
            print(f'\n\t~~~~shallowest layer with zero outputs = {sl}. Resampling...')# Resample param vals of layer.~~~')
            self.module_list[sl].resample_params()

        return

    def training_epoch_end(self, train_step_outputs):
        # this must be overridden for batch outputs to be fed to callback. Bug.
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4326

        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        self.log(name=f'train/{self.hparams.loss}_epoch', value=avg_loss, on_step=False, on_epoch=True)

        means, stdes = self.aggregate_step_outputs(outputs=train_step_outputs)
        self.log_metrics(means, stdes, stage='train')
        #self.log_glad_nmse(means=means, outputs=train_step_outputs, stage='train')
        return None

    def on_validation_start(self) -> None:
        # find best threshold only once before every validation epoch
        # use training set to optimize threshold during training
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=200)
        self.threshold = self.find_threshold(dl=self.trainer.datamodule.train_dataloader(), threshold_test_points=test_points, metric2chooseThresh=self.hparams.threshold_metric)
        #self.threshold[0] = new_threshold.clone().detach().view(1)
        #self.threshold[0] = new_threshold.clone().detach().view(1)
        self.log('threshold', self.threshold, prog_bar=True, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self.shared_step(batch)
        self.threshold = self.threshold.to(self.device)
        metrics = self.compile_metrics(y_hat=y_hat.detach(), y=y, threshold=self.threshold)
        return {'metrics': metrics, 'batch_size': len(x)}

    ###
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
            elif self.hparams.prior_construction in ['zeros', None]:
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
            return torch.zeros(size=(N, N), device=self.device).expand(prior_channels, batch_size, N, N)
        elif self.hparams.prior_construction == 'ones':
            return 0.5 * torch.ones(size=(N, N), device=self.device).expand(prior_channels, batch_size, N, N)
        elif self.hparams.prior_construction == 'block':
            block_scale = .35  # minimizes se for brain graphs
            assert (N % 2) == 0, f'block diagram requires even N'
            ones = torch.ones(size=((N // 2), (N // 2)), device=self.device)
            return torch.block_diag(ones, ones).expand(prior_channels, batch_size, N, N) * block_scale
        elif self.hparams.prior_construction in ['mean', 'median', 'sbm']:
            # must use training_prior
            assert self.training_prior.shape[-1] == N, f'when using prior constructed from data, we cannot test on data of different size than it'
            return self.training_prior.expand(batch_size, N, N) # (prior_channels, batch_size, N, N)

    def prior_performance(self, prior_channel, prior_channel_name, holdout_set_threshold, holdout_set_metrics):
        # find best threshold to use to optimize acc on train set
        # optimize threshold on training set
        y_hat_vec = adj2vec(prior_channel.expand(holdout_set_threshold.shape).detach())
        test_points = torch.linspace(start=y_hat_vec.min().item(), end=y_hat_vec.max().item(),
                                     steps=200) #self.hparams.num_threshold_discretizations)
        threshold = \
            best_threshold_by_metric(
                y_hat=y_hat_vec,
                y=adj2vec(holdout_set_threshold.detach()),
                thresholds=test_points,
                metric=self.hparams.threshold_metric,
                device=self.device
            )

        # find performance on validation set for each subnetwork using threshold found on train set
        y, y_hat = holdout_set_metrics.detach(), torch.broadcast_to(prior_channel, holdout_set_metrics.shape).detach()
        subnetwork_metrics = {}
        for subnetwork_name, subnetwork_mask in self.trainer.datamodule.subnetwork_masks.items(): #self.subnetwork_masks.items():
            y_subnet = adj2vec(apply_mask(y, subnetwork_mask))
            y_hat_subnet = adj2vec(apply_mask(y_hat.detach(), subnetwork_mask))
            subnetwork_metrics[subnetwork_name] = compute_metrics(y_hat=y_hat_subnet, y=y_subnet, threshold=threshold,
            #                                                      non_neg=self.trainer.datamodule.non_neg_labels,
                                                                  self_loops=self.trainer.datamodule.self_loops)
        # display metrics found
        print(f"Prior {prior_channel_name} metrics using {threshold}")
        print(f'ON VAL')
        # only interested in mean at the moment...TODO: print stde
        mean_metrics = {}
        for subnetwork, metric_dict in subnetwork_metrics.items():
            mean_metrics[subnetwork_name] = {}
            for metric_name, metric_values in metric_dict.items():
                mean_metrics[subnetwork_name][metric_name] = torch.mean(metric_values)
        # mean, stde = self.aggregate_step_outputs(outputs={'metrics': subnetwork_metrics['full']})
        # mean_metrics = {subnetwork: {metric_name: torch.mean(metric_values[metric_name])} for metric_name, metric_values in metrics.keys() for subnetwork, metrics in subnetwork_metrics.items()}
        print_subnet_perf_dict(subnetwork_metrics_dict=mean_metrics,  # {'full': mean_metrics},
                               indents=2, convert_to_percent=['acc', 'error'],
                               metrics2print=['se', 'ae', 'nmse', 'acc', 'error', 'mcc'])
        return threshold, mean_metrics  # subnetwork_metrics

    def compile_metrics(self, y_hat, y, threshold):
        # handles complexity of metrics with and without including the diagonal
        y_hat_vec = adj2vec(y_hat.detach())
        y_vec = adj2vec(y.detach())

        # REGRESSION METRICS: W/ & W/O Diag
        # compute regression metrics on off diagonal - doesn't care about signedness of elements
        off_diag_regress_metrics = regression_metrics(y_hat=y_hat_vec, y=y_vec, self_loops=False)
        rg_diag_metrics = regression_metrics(y_hat=y_hat.detach(), y=y, self_loops=True)  # NMSE
        rg_diag_metrics = {k + '-D': rg_diag_metrics[k] for k in rg_diag_metrics.keys()}
        metrics = {**off_diag_regress_metrics, **rg_diag_metrics}

        # CLASSIFICATION METRICS: HOMOGONEITY OF EDGE WEIGHT SIGNS
        # If all same sign (non-neg/non-pos) -> then can use all typical binary classifcation metrics
        # If not,
        # non_neg, self_loops = True, False

        if self.trainer.datamodule.label in ['adjacency', 'laplacian']:
            # all off diagonal elements are of the SAME sign.
            same_sign = True
            off_diag_class_metrics = symmetric_classification_metrics(y_hat=y_hat_vec, y=y_vec, threshold=threshold)
            metrics = {**metrics, **off_diag_class_metrics}
        else:
            raise ValueError('MUST HANDLE THIS CASE OF NON-HOMOG LABELS')

        return metrics
    ###

    def off_diag_metrics(self, batch, y_hat):
        # special case - compute metrics of undirected graph, non-neg, w/o self loop by rmeoving diag
        # find threshold, computes metrics only considering off diagonal entires
        x, y = batch[:2]
        print(f'label: {self.trainer.datamodule.label}')
        # remove off diagonal entries, make into vec
        y_, y_hat_ = adj2vec(y), adj2vec(y_hat)
        test_points = torch.linspace(start=y_hat_.min().item(), end=y_hat_.max().item(), steps=200)
        threshold = self.find_threshold(dl=self.trainer.datamodule.val_dataloader(), threshold_test_points=test_points, metric2chooseThresh=self.hparams.threshold_metric)
        metrics = compute_metrics(y_hat=y_hat_, y=y_, threshold=threshold, self_loops=False, non_neg=self.trainer.datamodule.non_neg_labels)
        print("normal_mle undirected non-neg metrics:", "nse: ", metrics['nse'].mean(), "se: ", metrics['se'].mean(),
              "error: ", metrics['error'].mean())

    def validation_epoch_end(self, val_step_outputs):
        means, stdes = self.aggregate_step_outputs(outputs=val_step_outputs)

        #self.log_glad_nmse(means=means, outputs=val_step_outputs, stage='val')


        # save running list of metrics as we go
        self.list_of_metrics.append({'means': means, 'stdes': stdes, 'epoch': self.current_epoch})
        self.log_metrics(means, stdes, stage='val')
        """
        stage = 'val'
        for metric_name in means.keys():
            name = f'{stage}/{metric_name}'
            self.log(name=name+'/'+'mean', value=means[metric_name])
            self.log(name=name+'/'+'stde', value=stdes[metric_name])
        """

        # for progress bar
        #if self.trainer.datamodule.non_neg_labels:
        #    metrics_in_progress_bar = ['se', 'se_per_edge', 'nse', '10_log_nse', 'error', 'mcc', 'f1']#'ae']
        #else:
        #    metrics_in_progress_bar = ['se', 'se_per_edge', 'nse', 'error']  # , 'mcc', 'f1']#'ae']
        metrics_in_progress_bar = ['se', 'se_per_edge', 'nse', 'error']  # , 'mcc', 'f1']#'ae']
        prog_bar_metrics_dict = {}
        for metric_name in metrics_in_progress_bar:
            prog_bar_metrics_dict[metric_name] = 100 * means[metric_name] if metric_name in ['acc', 'error', 'mcc', 'f1'] else means[metric_name]
            #prog_bar_metrics_dict[metric_name] = 100*means[metric_name] if metric_name not in ['nse', '10_log_nse', 'se', 'se_per_edge', 'ae', 'hinge' 'se_per_edge'] else means[metric_name]
        self.log_dict(prog_bar_metrics_dict, logger=False, prog_bar=True)

        # to able to see/log lr, need to do this
        current_lr = self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        self.log("lr", round(current_lr, 10), logger=True, prog_bar=True)

        # print summary of training results on full network: green good, red bad
        print(f'\nPerformance (using training set ONLY for train/threshold finding, and eval on validation) using loss: *{self.hparams.loss}')
        #if self.trainer.datamodule.non_neg_labels:
        #    logged_metrics = ['val/f1', 'val/error', 'val/ae', 'val/se', 'val/se_per_edge' 'val/nse']#, 'val/10log_nse']
        #else:
        #    logged_metrics = ['val/error', 'val/ae', 'val/se', 'val/nse', 'val/se_per_edge']#, 'val/10log_nse']
        logged_metrics = ['val/error', 'val/ae', 'val/se', 'val/nse', 'val/se_per_edge']#, 'val/10log_nse']

        for log_metric in logged_metrics:
            metric_name = log_metric.split("/")[-1]
            maximize = any(m in metric_name for m in ['acc', 'mcc', 'f1'])
            best_epoch = self.best_metrics(sort_metric=metric_name, maximize=maximize)[0]
            best_val, current_val = best_epoch['means'][metric_name], means[metric_name]
            print(f"{f'  {metric_name}: Best ':<15}", end="")
            print(f" {best_val:.5f} on epoch {best_epoch['epoch']}", end="")
            print(f" | Current: {current_val:.5f}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['list_of_metrics'] = self.list_of_metrics
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=200)
        self.testing_threshold = self.find_threshold(dl=self.trainer.datamodule.val_dataloader(),
                                                     threshold_test_points=test_points,
                                                     metric2chooseThresh=self.hparams.threshold_metric)
        checkpoint['testing_threshold'] = self.testing_threshold
        print(f"\n\tsaving checkpoint: saving threshold found {checkpoint['testing_threshold'].item():.3f} using validation set during training")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.list_of_metrics = checkpoint['list_of_metrics']
        #self.testing_threshold[0] = checkpoint['testing_threshold']
        self.testing_threshold = checkpoint['testing_threshold']
        self.testing_threshold.to(self.device)
        #self.register_buffer('testing_threshold', checkpoint['testing_threshold'], persistent=True)
        self.checkpoint_loaded = True
        print(f"\nLoading threshold found using validaiton set during training: {self.testing_threshold.item():.5f} which achieved {self.list_of_metrics[-1]['means']['error']*100:.4f}% error")

    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self.shared_step(batch)
        self.testing_threshold = self.testing_threshold.to(self.device)
        metrics = self.compile_metrics(y_hat=y_hat.detach(), y=y, threshold=self.testing_threshold)
        return {'metrics': metrics, 'batch_size': len(x)}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # LOG CALLBACK
        means, stdes = self.aggregate_step_outputs(outputs=outputs)
        self.log_metrics(means, stdes, stage='test')
        return None

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        if 'adam' in self.hparams.optimizer:
            b1, b2 = self.hparams.adam_beta_1, self.hparams.adam_beta_2
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2), eps=1e-08, weight_decay=0)
        elif 'sgd' in self.hparams.optimizer:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=self.hparams.momentum, dampening=0,
                                        weight_decay=0, nesterov=False)
        else:
            raise ValueError(f'only configured Adam and SGD optimizer. Given {self.hparams.optimizer}')

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 100, 200], gamma=0.25)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 100, 200], gamma=0.25)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=2*[10, 15, 20, 25, 100, 200], gamma=0.25)
        return {'optimizer': optimizer}#, 'lr_scheduler': {'scheduler': scheduler, 'monitor': self.hparams.monitor, 'frequency': 1}} #self.trainer.check_val_every_n_epoch}}

    ### HELPER METHODS ###
    @torch.no_grad()
    def aggregate_step_outputs(self, outputs):
        # aggregate all outputs from step batches
        total_epoch_samples = torch.stack([torch.tensor(x['batch_size']) for x in outputs]).sum()
        all_sample_metrics = {m: [] for m in outputs[0]['metrics'].keys()}
        for output in outputs:
            for metric_name, metric_values in output['metrics'].items():
                all_sample_metrics[metric_name].append(metric_values)

        # combine list of tensors into one large tensor
        for metric_name, metric_values in all_sample_metrics.items():
            all_sample_metrics[metric_name] = torch.cat(metric_values)
            """
            if 'glad' in metric_name:
                # will throw error bc cant concat single dim tensors
                all_sample_metrics[metric_name] = torch.tensor(metric_values)
            else:
                all_sample_metrics[metric_name] = torch.cat(metric_values)
            """

        # compute mean and standard error of each
        means, stdes = {}, {}
        for metric_name, metric_values in all_sample_metrics.items():
            means[metric_name] = torch.mean(metric_values)
            stdes[metric_name] = torch.std(metric_values) / math.sqrt(len(metric_values))

        return means, stdes

    # DOES NOT CHANGE ANY PRIVATE VARIABLES: STATIC METHOD
    @torch.no_grad()
    def find_threshold(self, dl, threshold_test_points, metric2chooseThresh):
        # use data (train or val) set to optimize threshold during training
        ys, y_hats = [], []
        for i, batch in enumerate(iter(dl)):  # loop so dont run out of memory
            batch[0] = batch[0].to(self.device)  # move fcs/scs to GPU (ligthning doesnt do this for us here)
            batch[1] = batch[1].to(self.device)
            ys.append(batch[1])
            y_hats.append(self.shared_step(batch))
        y, y_hat = torch.cat(ys, dim=0), torch.cat(y_hats, dim=0)
        if y.ndim == 3:
            y = adj2vec(y)
        if y_hat.ndim == 3:
            y_hat = adj2vec(y_hat)

        # loop over candidate thresholds, see which one optimizes threshold_metric (acc, mcc, se, etc)
        # over FULL network
        non_neg = True # self.trainer.datamodule.non_neg_labels
        if self.trainer.datamodule.label in ['adjacency', 'laplacian']:
            # all off diagonal elements are of the SAME sign.
            same_sign = True
        else:
            raise ValueError('HANDLE NON-HOMOG LABLES')

        threshold = best_threshold_by_metric(y_hat=torch.abs(y_hat).squeeze(),
                                             y=torch.abs(y).squeeze(),
                                             thresholds=threshold_test_points, metric=metric2chooseThresh, device=self.device)
        return threshold.view(1)

    def compute_loss(self, intermediate_outputs, y, use_raw_adj, per_sample_loss=False, per_edge_loss=False):
        # intermediate outputs INCLUDE final output
        #assert y.ndim == 2
        assert len(intermediate_outputs) == self.hparams.depth
        assert not (per_sample_loss and per_edge_loss), f'can only choose at most one loss normalization'
        batch_size, n = y.shape[:2]
        num_edges = n * (n-1) // 2
        total_edges = (batch_size * n * n) if use_raw_adj else batch_size * num_edges

        # naive method
        if not use_raw_adj:
            intermediate_outputs = [adj2vec(int_out) for int_out in intermediate_outputs]
            assert y.ndim == 2, f'assuming y is vectorized, error if adj'
            y = vec2adj(y, n=n)

        losses = torch.zeros(self.hparams.depth)
        for d, y_hat_i in enumerate(intermediate_outputs):
            # compute a loss for each intermediate output
            if self.hparams.loss == 'nse':
                # NOTE NORMALIZATION WILL BE WRONG FOR nmse -> WE ARE ALREADY COMPUTING MEAN (reduction of batch size), THEN WILL DIVIDE BY BS *N *N edges
                assert use_raw_adj
                reduction_dims = (1, 2) if use_raw_adj else 1
                se = ((y-y_hat_i)**2).sum(dim=reduction_dims)
                nse = se / (y ** 2).sum(dim=reduction_dims)
                loss = nse.sum()
                #loss = torch.divide(mse, torch.linalg.norm(y, ord=2, dim=reduction_dims)).sum()
            elif self.hparams.loss == 'se':
                loss = ((y - y_hat_i) ** 2).sum()
                #loss = F.mse_loss(y_hat_i, y, reduction='sum')
            elif self.hparams.loss == 'ae':
                loss = F.l1_loss(y_hat_i, y, reduction='sum')
            elif self.hparams.loss == 'hinge':
                assert self.trainer.datamodule.non_neg_labels, f'can only do hinge on non-negative edges!'
                loss = hinge_loss(y=y > 0, y_hat=y_hat_i, margin=self.hparams.hinge_margin,
                                  slope=self.hparams.hinge_slope, per_edge=False).sum()
            else:
                raise ValueError(f'loss {self.hparams.loss} not recognized')

            # weight the loss by depth of unrolling: weight loss more as we get closer to end
            depth_scaling = self.hparams.gamma ** (self.hparams.depth - (d+1)) / self.hparams.depth
            losses[d] = depth_scaling * loss

        if per_sample_loss:
            # averaged over samples in batch
            losses = losses / batch_size
        elif per_edge_loss:
            # averaged over each possible edge: N^2 for raw adj, N*(N-1)/2 for symm w/o self-loops
            losses = losses / total_edges

        return losses.sum()
        """
        optimized tensor version
        # intermediate outputs is a list, with entry i being a batch_size x N x N tensor.
        # 0) concatenate -> [bsxNxN,...,bsxNxN] -> depth x bs x N x N
        # 1) convert to appropriate diagonal form -> depth x bs x N(N-1)/2
        # 2) expand y to match shape
        # 3) weight by depth d in network ~ gamma^{D - d}
        i_outs = torch.cat(intermediate_outputs, dim=0)
        y_expand = y.repeat(self.hparams.depth, self.hparams.depth, 1, 1)
        assert i_outs.shape == y_expand.shape
        assert i_outs.shape == (self.hparams.depth, batch_size, n*(n-1)//2)
        weight = torch.tensor([self.hparams.gamma**(self.hparams.depth - d) for d in range(1, self.hparams.depth+1)]).view(self.hparams.depth, 1, 1)


        #if self.hparams.loss in ['hinge', 'cross entropy']:
        num_possible_edges = y.shape[-1]
        # sum_to_mean_constant = fcs.shape[0]*fcs.shape[1]*fcs.shape[2] - fcs.shape[1] #remove all zero diagonal from constant?? These will always be zero
        per_edge_loss = loss / (bs*num_possible_edges)
        return per_edge_loss
        #else:
        #    return loss
        """

    def best_metrics(self, sort_metric, top_k=1, maximize=True):
        sorted_list_of_metrics = sorted(self.list_of_metrics, key=lambda e: e['means'][sort_metric], reverse=maximize)
        return sorted_list_of_metrics[:top_k]

    def log_glad_nmse(self, means, outputs, stage):
        # NMSE is defined (perhaps incorrectly) as TOTAL squared error over TOTAL true label size

        # total squared error is average squared error * total number of samples
        total_epoch_samples = torch.stack([torch.tensor(x['batch_size']) for x in outputs]).sum()
        total_se = means['se']*total_epoch_samples

        # we cached total label size in outputs
        total_label_size = torch.tensor([x['label_size'] for x in outputs]).sum()

        glad_nmse = 10 * torch.log10(total_se / total_label_size)
        self.log(name='10log_nse', value=glad_nmse, prog_bar=stage == 'val')
        self.log(name=f'{stage}/10log_nse', value=glad_nmse, logger=True)

    def log_metrics(self, means, stdes, stage):

        for metric_name in means.keys():
            name = f'{stage}/{metric_name}'
            self.log(name=name + '/' + 'mean', value=means[metric_name])
            self.log(name=name + '/' + 'stde', value=stdes[metric_name])


if __name__ == "__main__":
    print('GLAD main loop')

