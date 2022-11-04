"""

    GLAD unrolls AM iterations on the GLASSO objective. It requires Positive Definiteness of intermedate graph outputs.
    It uses matrix square root operation. Exact matrix square root operations are expensive - in both the forward and
    backward pass - and we provide fast approximate alternatives for this which do not seem to degrade performance. See
    matrix_sqrt.py in this folder for more info.
    regression metrics must be updated to account for diagonal.
"""

from pytorch_lightning import seed_everything
path2currDir = str(Path.cwd()) + '/'
from graph_learning.misc.metrics import best_threshold_by_metric, compute_metrics, hinge_loss, regression_metrics, symmetric_classification_metrics
from graph_learning.models.unroll.glad.matrix_sqrt import MatrixSquareRoot
from graph_learning.misc.utils import adj2vec, vec2adj, adjs2fvs
sqrtm = MatrixSquareRoot.apply
        print_subnet_perf_dict, percent_change_metrics
except:

    #starting from beginning of model, check if the layer output all zeros.
    #  Return layer depth or -1
    for i, module in enumerate(model.module_list):

def rhoNN(f: int, h: int, dtype):
    # f = # input features

    # f = # input features
    # h = hidden layer size
                         nn.Linear(h, h, dtype=dtype), nn.Tanh(),
                         nn.Linear(h, 1, dtype=dtype), nn.Sigmoid())

    def __init__(self, h: int,  # hidden layer size
                       dtype: torch.dtype
        self.lambda_nn, self.rho_nn = lambdaNN(f=2, h=h, dtype=dtype), rhoNN(f=3, h=h, dtype=dtype)
    def forward(self, sigma: torch.tensor, theta: torch.tensor, z: torch.tensor, lambda_: torch.tensor, non_neg: bool):
        rho = self.rho_nn(fvs).view(batch_size, n, n) # (bs * n * n, 1) -> (bs, n, n)
        z = torch.sign(theta) * torch.relu(torch.abs(theta) - rho)
        z = torch.relu(z) if non_neg else z

        if torch.allclose(z, torch.zeros(1, device=z.device)) or \
                torch.allclose(theta, torch.zeros(1, device=theta.device)):
            self.output_zeros = True

        return theta, z, lambda_
                 # architecture
                 depth: int,  # number of unrolled iterations
                 h: int,  # hidden layer size
                 # loss, optimizer
                 optimizer: str = 'adam',
                 learning_rate: float = .03,
                 adam_beta_1: float = .9, adam_beta_2: float = .999, momentum: float = .95,
                 hinge_margin: float = .25, hinge_slope: float = 1.0,
                 gamma: float = 0.8,
                 # threshold
            seed_everything(seed, workers=True)
        #    assert n_train is not None, f'if using prior, must provide size of prior'
        # best threshold found on validation set after training done, to be used for testing
        self.register_buffer('testing_prior_threshold', -1 * torch.ones(num_prior_channels, 1), persistent=True)

        # self.register_buffer('real_data_prior', torch.zeros(1, 68, 68), persistent=True)
        # self.register_buffer('test_threshold', torch.tensor([-1.0]), persistent=True)
        ######
        self.checkpoint_loaded = False  # are we being loaded from a checkpoint?


        # best threshold found on training set when training_prior was constructed
        self.register_buffer('threshold', torch.tensor([-2.0]), persistent=True)

        # best threshold found on validation set after training done, to be used for testing
        self.min_output = torch.tensor([-1])
        if share_parameters:
            #self.channels[0] = 1
            for layer in range(depth):

        # dont use prior_channels/names yet
            # assert torch.allclose(self.training_prior, -1*torch.ones(1)), f"init'ed to all -1. This is how we know not used yet."
            self.prior_channels, self.prior_channel_names = self.construct_prior_(prior_dl=self.trainer.datamodule.train_dataloader())
            try:
                train_scs, val_scs = self.trainer.datamodule.train_dataloader().dataset.full_ds()[1], \
                                     self.trainer.datamodule.val_dataloader().dataset.full_ds()[1]
            except AttributeError:  # AttributeError: 'Subset' object has no attribute 'full_ds'
                self.training_prior_threshold[i], self.prior_metrics_val[prior_channel_name] = \
                    self.prior_performance(prior_channel=prior_channel, prior_channel_name=prior_channel_name,
                                           holdout_set_threshold=train_scs, holdout_set_metrics=val_scs)
            """
    def forward(self, batch) -> Any:
        return self.shared_step(batch=batch)

        x, y = batch[:2]

        I = torch.eye(n, device=self.device).expand(batch_size, n, n)
        if self.hparams.prior_construction is not None:
        z = theta.detach().clone()
        intermediate_outputs = []
        return (z, intermediate_outputs) if int_out else z


        loss = self.compute_loss(intermediate_outputs=intermediate_outputs, y=y, use_raw_adj=True, per_sample_loss=True)
        self.threshold = self.threshold.to(self.device)
        metrics = self.compile_metrics(y_hat=y_hat.detach(), y=y, threshold=self.threshold)

        return {'loss': loss, 'metrics': metrics, 'batch_size': len(x)}

    def on_after_backward(self) -> None:
            for name, param in m.named_parameters():
                if param.requires_grad:
                    name = name + ('' if self.hparams.share_parameters else f'_{i}')
            if param.requires_grad:
                self.log(name='param/value/' + name, value=param.data)
            if param.grad is not None:
                self.log(name='param/grad/' + name, value=param.grad)

        """
        # find the shallowest layer where model outputting ALL zeros. -1 if no layer outputting all zeros.
        sl = shallowest_layer_all_zero(self)

        if sl > -1:
            #sl = shallowest_layer_all_zero(self)
            print(f'\n\t~~~~shallowest layer with zero outputs = {sl}. Resampling...')# Resample param vals of layer.~~~')
            self.module_list[sl].resample_params()

        return
        # this must be overridden for batch outputs to be fed to callback. Bug.
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4326

        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        self.log(name=f'train/{self.hparams.loss}_epoch', value=avg_loss, on_step=False, on_epoch=True)

        means, stdes = self.aggregate_step_outputs(outputs=train_step_outputs)
        self.log_metrics(means, stdes, stage='train')
        #self.log_glad_nmse(means=means, outputs=train_step_outputs, stage='train')
    def on_validation_start(self) -> None:
        # find best threshold only once before every validation epoch
        # use training set to optimize threshold during training
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=200)
        self.threshold = self.find_threshold(dl=self.trainer.datamodule.train_dataloader(), threshold_test_points=test_points, metric2chooseThresh=self.hparams.threshold_metric)
        #self.threshold[0] = new_threshold.clone().detach().view(1)
        #self.threshold[0] = new_threshold.clone().detach().view(1)
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
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
                prior_channels = [self.training_prior[0, 0], self.training_prior[1, 0]]
        # This is useful when we want to test on different sized data than we trained on: simply feed in test set as
        # val_dl, and prior will be constructed appropriately.
        else:
            if self.hparams.prior_construction == 'block':
                block_scale = .35  # minimizes se
                # self.prior = torch.block_diag(ones, ones).view(1, N, N)*block_scale
                # for sc in np.arange(0.3, .4, .01):
                #    print(f'scale: {sc}', predicition_metrics(y_hat=sc*torch.block_diag(ones, ones).view(1, N, N).repeat(len(val_scs), 1, 1), y=val_scs, y_subject_ids=val_subject_ids))
                prior_channel_names = ['sbm']
            elif self.hparams.prior_construction in ['zeros', None]:
                # self.prior = torch.zeros(1, N, N, dtype=train_fcs.dtype)
                prior_channels = [torch.zeros(1, self.hparams.n_train, self.hparams.n_train, dtype=prior_dtype)]
                prior_channel_names = ['zeros']
            elif self.hparams.prior_construction in ['ones']:
                prior_channels = [torch.ones(1, self.hparams.n_train, self.hparams.n_train, dtype=prior_dtype)]
                prior_channel_names = ['ones']
            self.training_prior[0] = prior_channels[0]

    def prior_prep(self, batch_size, N):
        # called by train_step/val_step/test_step to construct the prior tensor.

        # By NOT using self.training_prior (constructed in setup() during initial training) when we don't
        # need to (e.g. for mean/median), then we can test on graphs of sizes different than those trained on.
        prior_channels = 1 #if (not self.hparams.share_parameters) else self.layers[0].c_in
        if self.hparams.prior_construction == 'zeros':
            return torch.zeros(size=(N, N), device=self.device).expand(prior_channels, batch_size, N, N)
        elif self.hparams.prior_construction == 'ones':
            block_scale = .35  # minimizes se for brain graphs
            assert (N % 2) == 0, f'block diagram requires even N'
            ones = torch.ones(size=((N // 2), (N // 2)), device=self.device)
            assert self.training_prior.shape[-1] == N, f'when using prior constructed from data, we cannot test on data of different size than it'
            return self.training_prior.expand(batch_size, N, N) # (prior_channels, batch_size, N, N)

    def prior_performance(self, prior_channel, prior_channel_name, holdout_set_threshold, holdout_set_metrics):
        y_hat_vec = adj2vec(prior_channel.expand(holdout_set_threshold.shape).detach())
                y_hat=y_hat_vec,
                y=adj2vec(holdout_set_threshold.detach()),
                thresholds=test_points,
                metric=self.hparams.threshold_metric,
            y_subnet = adj2vec(apply_mask(y, subnetwork_mask))
            y_hat_subnet = adj2vec(apply_mask(y_hat.detach(), subnetwork_mask))
                                                                  self_loops=self.trainer.datamodule.self_loops)
        # display metrics found
        print(f"Prior {prior_channel_name} metrics using {threshold}")
        print(f'ON VAL')
        # only interested in mean at the moment...TODO: print stde
        mean_metrics = {}
        for subnetwork, metric_dict in subnetwork_metrics.items():
                mean_metrics[subnetwork_name][metric_name] = torch.mean(metric_values)

        y_vec = adj2vec(y.detach())

        # REGRESSION METRICS: W/ & W/O Diag
        metrics = {**off_diag_regress_metrics, **rg_diag_metrics}
        # If all same sign (non-neg/non-pos) -> then can use all typical binary classifcation metrics
        # If not,
        # non_neg, self_loops = True, False

        if self.trainer.datamodule.label in ['adjacency', 'laplacian']:
            # all off diagonal elements are of the SAME sign.
            metrics = {**metrics, **off_diag_class_metrics}
        else:
        return metrics
    ###

        # find threshold, computes metrics only considering off diagonal entires
        y_, y_hat_ = adj2vec(y), adj2vec(y_hat)
        test_points = torch.linspace(start=y_hat_.min().item(), end=y_hat_.max().item(), steps=200)
        threshold = self.find_threshold(dl=self.trainer.datamodule.val_dataloader(), threshold_test_points=test_points, metric2chooseThresh=self.hparams.threshold_metric)
        metrics = compute_metrics(y_hat=y_hat_, y=y_, threshold=threshold, self_loops=False, non_neg=self.trainer.datamodule.non_neg_labels)
        print("normal_mle undirected non-neg metrics:", "nse: ", metrics['nse'].mean(), "se: ", metrics['se'].mean(),
              "error: ", metrics['error'].mean())
        means, stdes = self.aggregate_step_outputs(outputs=val_step_outputs)

        #self.log_glad_nmse(means=means, outputs=val_step_outputs, stage='val')


        # save running list of metrics as we go
        self.list_of_metrics.append({'means': means, 'stdes': stdes, 'epoch': self.current_epoch})
        stage = 'val'
        for metric_name in means.keys():
            name = f'{stage}/{metric_name}'
            self.log(name=name+'/'+'mean', value=means[metric_name])
            self.log(name=name+'/'+'stde', value=stdes[metric_name])
        #    metrics_in_progress_bar = ['se', 'se_per_edge', 'nse', 'error']  # , 'mcc', 'f1']#'ae']
        metrics_in_progress_bar = ['se', 'se_per_edge', 'nse', 'error']  # , 'mcc', 'f1']#'ae']
        prog_bar_metrics_dict = {}

        print(f'\nPerformance (using training set ONLY for train/threshold finding, and eval on validation) using loss: *{self.hparams.loss}')
        #if self.trainer.datamodule.non_neg_labels:
        #    logged_metrics = ['val/f1', 'val/error', 'val/ae', 'val/se', 'val/se_per_edge' 'val/nse']#, 'val/10log_nse']
        #else:
        #    logged_metrics = ['val/error', 'val/ae', 'val/se', 'val/nse', 'val/se_per_edge']#, 'val/10log_nse']
            maximize = any(m in metric_name for m in ['acc', 'mcc', 'f1'])
            best_epoch = self.best_metrics(sort_metric=metric_name, maximize=maximize)[0]
            best_val, current_val = best_epoch['means'][metric_name], means[metric_name]
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['list_of_metrics'] = self.list_of_metrics
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=200)
                                                     metric2chooseThresh=self.hparams.threshold_metric)
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.testing_threshold = checkpoint['testing_threshold']
        self.testing_threshold.to(self.device)
        print(f"\nLoading threshold found using validaiton set during training: {self.testing_threshold.item():.5f} which achieved {self.list_of_metrics[-1]['means']['error']*100:.4f}% error")
        y_hat = self.shared_step(batch)
        self.testing_threshold = self.testing_threshold.to(self.device)
        metrics = self.compile_metrics(y_hat=y_hat.detach(), y=y, threshold=self.testing_threshold)
        return None
        lr = self.hparams.learning_rate
        if 'adam' in self.hparams.optimizer:
        elif 'sgd' in self.hparams.optimizer:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=self.hparams.momentum, dampening=0,
        return {'optimizer': optimizer}#, 'lr_scheduler': {'scheduler': scheduler, 'monitor': self.hparams.monitor, 'frequency': 1}} #self.trainer.check_val_every_n_epoch}}

    ### HELPER METHODS ###
        all_sample_metrics = {m: [] for m in outputs[0]['metrics'].keys()}
        for output in outputs:
            for metric_name, metric_values in output['metrics'].items():
        # combine list of tensors into one large tensor
        for metric_name, metric_values in all_sample_metrics.items():
            all_sample_metrics[metric_name] = torch.cat(metric_values)
            """
            if 'glad' in metric_name:
                # will throw error bc cant concat single dim tensors
                all_sample_metrics[metric_name] = torch.tensor(metric_values)

        # compute mean and standard error of each
        means, stdes = {}, {}
            stdes[metric_name] = torch.std(metric_values) / math.sqrt(len(metric_values))

    def find_threshold(self, dl, threshold_test_points, metric2chooseThresh):
        # use data (train or val) set to optimize threshold during training
        ys, y_hats = [], []
        for i, batch in enumerate(iter(dl)):  # loop so dont run out of memory
            batch[0] = batch[0].to(self.device)  # move fcs/scs to GPU (ligthning doesnt do this for us here)
            batch[1] = batch[1].to(self.device)
            ys.append(batch[1])
        if y.ndim == 3:
            y = adj2vec(y)
        if y_hat.ndim == 3:
            y_hat = adj2vec(y_hat)
        if self.trainer.datamodule.label in ['adjacency', 'laplacian']:
            # all off diagonal elements are of the SAME sign.

        #assert y.ndim == 2
        assert len(intermediate_outputs) == self.hparams.depth
        assert not (per_sample_loss and per_edge_loss), f'can only choose at most one loss normalization'
        batch_size, n = y.shape[:2]
        num_edges = n * (n-1) // 2
        total_edges = (batch_size * n * n) if use_raw_adj else batch_size * num_edges

        # naive method
        if not use_raw_adj:
            intermediate_outputs = [adj2vec(int_out) for int_out in intermediate_outputs]
        losses = torch.zeros(self.hparams.depth)
        for d, y_hat_i in enumerate(intermediate_outputs):
            # compute a loss for each intermediate output
            if self.hparams.loss == 'nse':
                loss = nse.sum()
                #loss = torch.divide(mse, torch.linalg.norm(y, ord=2, dim=reduction_dims)).sum()
                #loss = F.mse_loss(y_hat_i, y, reduction='sum')
            elif self.hparams.loss == 'hinge':
                assert self.trainer.datamodule.non_neg_labels, f'can only do hinge on non-negative edges!'
            else:
            # weight the loss by depth of unrolling: weight loss more as we get closer to end
            depth_scaling = self.hparams.gamma ** (self.hparams.depth - (d+1)) / self.hparams.depth
        elif per_edge_loss:
            # averaged over each possible edge: N^2 for raw adj, N*(N-1)/2 for symm w/o self-loops
import torch
from torch.autograd import Function, Variable
from graph_learning.models.unroll.glad.matrix_sqrt_utils import sqrt_newton_schulz, lyap_newton_schulz


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
          for excellent discussion of this, see page 4 of http://vis-www.cs.umass.edu/bcnn/docs/improved_bcnn.pdf
    """
    @staticmethod
    def forward(ctx, input):
        # Newton Schulz iterations for forward step
        sA, error = sqrt_newton_schulz(input, numIters=10, dtype=input.dtype)
        ctx.save_for_backward(sA)
        return sA

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients by iterative Lyapunov solver
        z, = ctx.saved_tensors
        dlda = lyap_newton_schulz(z, dldz=grad_output, numIters=10, dtype=grad_output.dtype)
        return dlda

# unclear why this is done
sqrtm = MatrixSquareRoot.apply


def tests():
    from matrix_sqrt_utils import create_symm_matrix
    from torch.autograd import gradcheck

    # perform reccomended step to ensure gradients found by finite-difference method matches those returned
    # from backward()
    a = create_symm_matrix(batchSize=2, dim=3, numPts=5, tau=1.0, dtype=torch.double)
    a = a.clone().detach().requires_grad_(True)# torch.tensor(a, requires_grad=True)
    assert gradcheck(MatrixSquareRoot.apply, a)

    # sanity test 1
    dldz = torch.rand(2, 3, 3).requires_grad_(True) # this is what would be returned from autograd
    a = a_sqrt = torch.eye(3).expand(2, 3, 3).requires_grad_(True) # this is from the forward prop
    z = sqrtm(a)
    a.retain_grad()
    z.backward(dldz) # argument to backward is what auto-grad would feed back to it: dldz.
    dlda = a.grad
    assert torch.allclose(0.5*dldz, dlda), f'when input a is idenitity, dlda_sqrt == 0.5 * dldz'



"matrix_sqrt.py" [noeol] 52L, 1965B
