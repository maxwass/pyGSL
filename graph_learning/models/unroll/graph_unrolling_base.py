"""
    Unrolling Mixin.

    If new unrolling implements required methods and have required varaibles (specified below), then this mixin handles
    loss computation, logging of metrics and parameter distributions, threshold finding for link-predicition, and more.

"""

import math, sys, torch

from torch import nn
from typing import Any, Dict, List, Optional, Type

from pathlib import Path
file = Path(__file__).resolve()
path2project = str(file.parents[3]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/

from graph_learning.misc.utils import adj2vec, vec2adj
from graph_learning.misc.metrics import best_threshold_by_metric, compute_metrics, hinge_loss
from graph_learning.models.unroll.gdn.gdn_utils import filter_repeats, construct_prior


class unrolling(nn.Module):
    """Requires methods: shared_step,  """
    """Requires attributes: max_output, min_output, list_of_metrics, threshold, testing_threshold, """
    def __init__(self):
        super().__init__()

    def forward(self, batch, **kwargs) -> Any:
        return self.shared_step(batch=batch, **kwargs)['prediction']

    def intermediate_outputs(self, batch):
        return self.shared_step(batch)['intermediate_outputs']

    def training_step(self, batch, batch_idx):
        return self.shared_step_(batch, batch_idx, stage='train')

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        # find the shallowest layer where model outputting ALL zeros. -1 if no layer outputting all zeros.
        #sl = self.shallowest_layer_all_zero()
        sl = self.shallowest_layer_all_zero(self.module_list)
        # add check for shallowst NAN!
        if sl > -1:
            print(f'\n\t~~~~shallowest layer with zero outputs = {sl}. Resampling...')# Resample param vals of layer.~~~')
            self.module_list[sl].resample_params()

        return

    def training_epoch_end(self, train_step_outputs):
        # this must be overridden for batch outputs to be fed to callback. Bug.
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4326
        self.shared_epoch_end(step_outputs=train_step_outputs, stage='train')

        return None

    def on_train_start(self):
        try:
            # not all methods have these implemented yet
            self.log_parameter_distrib()
            self.log_layer_param_hist()
        except:
            print('unable to log parameter distributions')

    def on_validation_start(self) -> None:
        # find best threshold only once before every validation epoch
        # use training set to optimize threshold during training
        test_points = torch.linspace(start=min(0.0, self.min_output.item()), end=self.max_output.item(), steps=self.hparams.num_threshold_test_points, device=self.device)
        self.threshold = self.find_threshold(dl=self.trainer.datamodule.train_dataloader(), threshold_test_points=test_points, metric2chooseThresh=self.hparams.threshold_metric)
        self.log('threshold', self.threshold, prog_bar=True, on_epoch=True, on_step=False)

        try:
            # not all methods have these implemented yet
            self.log_parameter_distrib()
            self.log_layer_param_hist()
        except:
            print('unable to log parameter distributions')

    def validation_step(self, batch, batch_idx):
        return self.shared_step_(batch, batch_idx, stage='val')

    def validation_epoch_end(self, val_step_outputs):
        means, stdes = self.shared_epoch_end(step_outputs=val_step_outputs, stage='val')

        # Update Progress Bar
        metrics_in_progress_bar = ['nse', 'se', 'error', 'mcc', 'f1'] # auc
        prog_bar_metrics_dict = {}
        for metric_name in metrics_in_progress_bar:
            prog_bar_metrics_dict[metric_name] = 100 * means[metric_name] if metric_name in ['acc', 'error', 'mcc', 'f1'] else means[metric_name]
        self.log_dict(prog_bar_metrics_dict, logger=False, prog_bar=True)

        # to able to see/log lr, need to do this
        current_lr = self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        self.log("lr", round(current_lr, 10), logger=True, prog_bar=True)

        # print summary of training results on full network: green good, red bad
        #print(f'\nPerformance (using training set ONLY for train/threshold finding, and eval on validation) using loss: *{self.hparams.loss}')
        print(f'\nValidation performance using loss: *{self.hparams.loss}')
        logged_metrics = ['val/error', 'val/ae', 'val/se', 'val/nse', 'val/data_loss'] # 'val/se_per_edge',

        for log_metric in logged_metrics:
            metric_name = log_metric.split("/")[-1]
            maximize = any(m in metric_name for m in ['acc', 'mcc', 'f1'])
            best_epoch = self.best_metrics(sort_metric=metric_name, maximize=maximize)[0]
            best_val, current_val = best_epoch['means'][metric_name], means[metric_name]
            print(f"{f'  {metric_name}: Best ':<15}", end="")
            print(f" {best_val:.5f} on epoch {best_epoch['epoch']}", end="")
            print(f" | Current: {current_val:.5f}")

    def test_step(self, batch, batch_idx):
        return self.shared_step_(batch, batch_idx, stage='test')

    def test_epoch_end(self, outputs: List[Any]) -> None:
        means, stdes = self.shared_epoch_end(step_outputs=outputs, stage='test')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['list_of_metrics'] = self.list_of_metrics
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=self.hparams.num_threshold_test_points, device=self.device)
        self.testing_threshold = self.find_threshold(dl=self.trainer.datamodule.val_dataloader(),
                                                     threshold_test_points=test_points,
                                                     metric2chooseThresh=self.hparams.threshold_metric)
        checkpoint['testing_threshold'] = self.testing_threshold
        print(f"\n\tsaving checkpoint: saving threshold found {checkpoint['testing_threshold'].item():.3f} using validation set during training")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.list_of_metrics = checkpoint['list_of_metrics']
        self.testing_threshold = checkpoint['testing_threshold']
        self.testing_threshold.to(self.device)
        print(f"\nLoading threshold found using val set during training: {self.testing_threshold.item():.5f} which achieved {self.list_of_metrics[-1]['means']['error']*100:.4f}% error")

    # Helper methods #
    def shared_epoch_end(self, step_outputs, stage):
        means, stdes = self.aggregate_step_outputs(outputs=step_outputs)
        self.log_metrics(means, stdes, stage=stage)

        if 'test' not in stage:
            losses = torch.tensor([out['loss'] for out in step_outputs])
            data_loss = torch.tensor([out['data_loss'] for out in step_outputs])
            reg_loss = torch.tensor([out['reg_loss'] for out in step_outputs])
            self.log(f"{stage}/loss_mean", losses.mean());
            self.log(f"{stage}/loss_stde", torch.std(losses) / math.sqrt(len(losses)))
            self.log(f"{stage}/data_loss_mean", data_loss.mean());
            self.log(f"{stage}/data_loss_stde", torch.std(data_loss) / math.sqrt(len(data_loss)))
            self.log(f"{stage}/reg_loss_mean", reg_loss.mean());
            self.log(f"{stage}/reg_loss_stde", torch.std(reg_loss) / math.sqrt(len(reg_loss)))

        if 'val' in stage:
            # save running list of metrics as we go after each val step
            self.list_of_metrics.append({'means': means, 'stdes': stdes, 'epoch': self.current_epoch})
        return means, stdes

    def shared_step_(self, batch, batch_idx, stage):
        """
            shared code between the training/validation/testing_step() methods
            NOT shared_step(). That is written by the particular model.
        """
        assert stage in ['train', 'val', 'test']
        x, y = batch[:2]
        out = self.shared_step(batch)
        y_hat, intermediate_outputs = out['prediction'], out['intermediate_outputs']
        if 'train' in stage:
            self.max_output = y_hat.max() if not torch.isnan(y_hat.max()) else torch.tensor([100])  # IGNORE DIAG IN MAX SEARCH
            self.min_output = y_hat.min() if not torch.isnan(y_hat.min()) else torch.tensor([-100])


        # compute_loss requires y and y_hat be vectorized
        y = adj2vec(y) if (y.ndim == 3) else y
        y_hat = adj2vec(y_hat) if (y_hat.ndim == 3) else y_hat

        assert self.hparams.loss == 'se', 'for now, loss must be SE not NSE'
        # compute_loss may have to be adapted for non-vectorized (GDN) outputs
        data_loss = self.compute_loss(intermediate_outputs=intermediate_outputs, y=y, y_hat=y_hat)
        l2_params = (self.unrolling_trainable_parameters() ** 2).sum()
        reg_loss = self.hparams.l2_reg * l2_params
        loss = data_loss + reg_loss

        # L2G output vectorized. Some data is not vectorized
        y_hat = vec2adj(y_hat, n=y.shape[-1]) if y.ndim == 3 else y_hat

        threshold = self.testing_threshold if stage == 'test' else self.threshold
        threshold = threshold.to(y_hat.device) # we will threshold y_hat with threshold
        metrics = compute_metrics(y_hat=y_hat.detach(), y=y, threshold=threshold, self_loops=False)
        return {'loss': loss, 'data_loss': data_loss, 'reg_loss': reg_loss, 'l2_params': l2_params,
                'metrics': metrics, 'batch_size': len(x)}

    @torch.no_grad()
    def aggregate_step_outputs(self, outputs):
        # aggregate all outputs from step batches
        total_epoch_samples = torch.stack([torch.tensor(x['batch_size']) for x in outputs]).sum()

        # Compile metrics over all graphs for all batches, e.g. each of the 5 batches has 50 samples and each
        #   sample has an accuracy -> compile all 250 accuracies together
        metric_names_of_interest = list(outputs[0]['metrics'].keys())
        # this brings together in single list e.g. the losses from each batch, the accuracies for ALL GRAPHs in each batch, etc
        all_sample_metrics = {m: [] for m in metric_names_of_interest}
        for output in outputs:
            for metric_name, metric_values in output['metrics'].items():
                all_sample_metrics[metric_name].append(metric_values)

        # combine list of tensors into one large tensor
        for metric_name, metric_values in all_sample_metrics.items():
            all_sample_metrics[metric_name] = torch.cat(metric_values)

        # compute mean and standard error of each
        means, stdes = {}, {}
        for metric_name, metric_values in all_sample_metrics.items():
            means[metric_name] = torch.mean(metric_values)
            stdes[metric_name] = torch.std(metric_values) / math.sqrt(len(metric_values))


        ## Do the same for the losses
        loss_names_of_interest = ['data_loss', 'reg_loss']
        all_batch_losses = {m: [] for m in loss_names_of_interest}
        for output in outputs:
            for name in loss_names_of_interest:
                all_batch_losses[name].append(output[name].unsqueeze(0)) # unsqueeze so not 0-dim -> cannot concat

        # combine list of tensors into one large tensor
        for name, values in all_batch_losses.items():
            all_batch_losses[name] = torch.cat(values)

        # compute mean and standard error of each
        for name, values in all_batch_losses.items():
            means[name] = values.mean()
            stdes[name] = torch.std(metric_values) / math.sqrt(len(metric_values))


        return means, stdes

    @torch.no_grad()
    def find_threshold(self, dl, threshold_test_points, metric2chooseThresh):
        # use data (train or val) set to optimize threshold during training
        ys, y_hats = [], []
        for i, batch in enumerate(iter(dl)):  # loop so dont run out of memory
            batch[0] = batch[0].to(self.device)  # move fcs/scs to GPU (ligthning doesnt do this for us here)
            batch[1] = batch[1].to(self.device)
            ys.append(batch[1])
            y_hats.append(self.shared_step(batch)['prediction'])
            #y_hats.append(self.forward(batch))
        if ys[-1].ndim == 2:
            y = torch.cat([y.transpose(-1, -2) for y in ys], dim=1).transpose(-1, -2)
            y_hat = torch.cat([y_h.transpose(-1, -2) for y_h in y_hats], dim=1).transpose(-1, -2)
        else:
            y, y_hat = torch.cat(ys, dim=0), torch.cat(y_hats, dim=0)

        # both y and y_hat need to be vectorized
        #y = adj2vec(y) if (y_hat.ndim == 2 and y.ndim == 3) else y
        #y_hat = adj2vec(y_hat) if y_hat.ndim == 3 else y_hat
        y = adj2vec(y) if y.ndim == 3 else y
        y_hat = adj2vec(y_hat) if y_hat.ndim == 3 else y_hat
        # loop over candidate thresholds, see which one optimizes threshold_metric (acc, mcc, se, etc)
        # over FULL network
        if not (self.trainer.datamodule.label in ['adjacency', 'laplacian']):
            print(f'think more deeply about handling case where edges can be different signs')
            raise ValueError('HANDLE NON-HOMOG LABELS')

        assert y.shape == y_hat.shape
        single_sample_batch = (y_hat.ndim == 2) and (y_hat.shape[0] == 1)
        return best_threshold_by_metric(y_hat=y_hat.abs().squeeze() if not single_sample_batch else y_hat.abs(),
                                        y=y.abs().squeeze() if not single_sample_batch else y,
                                        thresholds=threshold_test_points, metric=metric2chooseThresh, device=self.device).view(1)

    def compute_loss(self, intermediate_outputs, y, y_hat=None):
        assert y.ndim == y_hat.ndim == 2, f'must adjust for square outputs. Assume symmetric so simply vectorize'
        if intermediate_outputs is None:
            assert y_hat is not None, f'no intermed output given, need outputs'
            batch_size, num_edges = y_hat.shape
            assert y_hat.ndim == 2, f'only implemented for half'
            # nse for each intermediate output. last dimension are edge weights -> -1
            if self.hparams.loss == 'nse':
                loss = torch.sum((y_hat - y) ** 2, dim=-1) / torch.sum(y ** 2, dim=-1)
            elif self.hparams.loss == 'se':
                loss = torch.sum((y_hat - y) ** 2, dim=-1)
            elif self.hparams.loss == 'hinge':
                loss = hinge_loss(y=y, y_hat=y_hat, margin=.25, slope=1, per_edge=False)
            elif self.hparams.loss == 'BCE':
                assert False, f'NOT DEBUGGED YET!'
                loss = torch.nn.BCELoss(input=y_hat, target=y)
            else:
                raise ValueError(f'loss {self.hparams.loss} not recognized')
            return loss.mean()

        else:
            # generalize to have intermediate losses when intermediate outputs are available -> non-MIMO
            assert intermediate_outputs.ndim == 3
            depth, batch_size, num_edges = intermediate_outputs.shape
            #y_non_batch = y.clone()
            y = y.expand(depth, batch_size, num_edges)

            # nse for each intermediate output. last dimension are edge weights -> -1
            if self.hparams.loss == 'nse':
                intermed_error = torch.sum( (intermediate_outputs-y)**2, dim=-1) / torch.sum(y**2, dim=-1)
            elif self.hparams.loss == 'se':
                intermed_error = torch.sum( (intermediate_outputs-y)**2, dim=-1)
            elif self.hparams.loss == 'ae':
                intermed_error = torch.sum((intermediate_outputs - y).abs(), dim=-1) #reduction='sum')
            else:
                raise ValueError(f'loss {self.hparams.loss} not recognized')
            if self.hparams.intermed_loss_discount in [False, None, 'None'] or (self.hparams.intermed_loss_discount < 1e-10):
                # dont use intermediate losses -> all intermediate factors are 0 except final
                factor = torch.zeros(depth, device=self.device)
                factor[-1] = 1
            else:
                assert 0.0 < self.hparams.intermed_loss_discount < 1, f'intermed_loss_discount is {self.hparams.intermed_loss_discount}, doesnt make sense'
                # scaled by factor proportional to how deep it is: larger scaling for outputs closer to final output
                factor = torch.tensor([self.hparams.intermed_loss_discount ** i for i in range(depth, 0, -1)], device=self.device)
            factor = factor.unsqueeze(dim=-1).expand(depth, batch_size)
            # sum up all intermediate losses and average over batch
            loss = (intermed_error * factor).sum(dim=0).mean()

            return loss

    def best_metrics(self, sort_metric, top_k=1, maximize=True):
        sorted_list_of_metrics = sorted(self.list_of_metrics, key=lambda e: e['means'][sort_metric], reverse=maximize)
        return sorted_list_of_metrics[:top_k]

    def log_metrics(self, means, stdes, stage):

        for metric_name in means.keys():
            name = f'{stage}/{metric_name}'
            self.log(name=name + '/' + 'mean', value=means[metric_name])
            self.log(name=name + '/' + 'stde', value=stdes[metric_name])

    @staticmethod
    def shallowest_layer_all_zero(module_list):
        # starting from beginning of model, check if the layer output all zeros.
        #  Return layer depth or -1
        for i, module in enumerate(module_list):
            if module.output_zeros:
                print(f"\t\t{i}th layer!")
                return i
        return -1

    ### prior funcs to be replaced... ###
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

    def prior_prep(self, batch_size, N, vectorized):
        # called by train_step/val_step/test_step to construct the prior tensor.

        # By NOT using self.training_prior (constructed in setup() during initial training) when we don't
        # need to (e.g. for mean/median), then we can test on graphs of sizes different than those trained on.
        prior_channels = 1 #if (not self.hparams.share_parameters) else self.layers[0].c_in
        if self.hparams.prior_construction == 'zeros':
            p = torch.zeros(size=(N, N), device=self.device)
            if vectorized:
                p = adj2vec(p.view(1, N, N))
                num_edges = p.shape[-1]
                # need to impliment mimo
                #return p.expand(prior_channels, batch_size, num_edges)
                return p.expand(batch_size, num_edges)
            else:
                #return p.expand(prior_channels, batch_size, N, N)
                return p.expand(batch_size, N, N)
            #return torch.zeros(size=(N, N), device=self.device).expand(prior_channels, batch_size, N, N)
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


if __name__ == "__main__":
    print('unrolling mixin main loop')

