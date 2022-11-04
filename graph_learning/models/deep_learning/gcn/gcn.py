import math, sys, numpy as np, torch, pytorch_lightning as pl
from torch import nn
from pytorch_lightning import seed_everything
from typing import Any, Dict, List, Optional, Type

import graph_learning.misc.utils
from graph_learning.misc.utils import adj2vec, vec2adj, normalize_slices, mimo_tensor_polynomial, adjs2fvs
from graph_learning.misc.metrics import best_threshold_by_metric, hinge_loss

from graph_learning.misc.metrics import compute_metrics


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out, bias=False)
        torch.nn.init.xavier_normal_(self.projection.weight, gain=1)

    def forward(self, node_feats, adj_matrix):
        """
        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """

        # A * X * W
        node_feats = self.projection(node_feats) # X * W
        node_feats = torch.bmm(adj_matrix, node_feats) # A * X * W
        return node_feats

bce_loss = torch.nn.BCELoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')
class gcn(pl.LightningModule):
    def __init__(self,
                 # architecture
                 channels: List[int],
                 num_one_hot: int = 0,
                 which_loss: str = 'BCE', #hinge',
                 learning_rate: float = .01,
                 # reproducability
                 seed: int = 50,
                 # threshold finding
                 threshold_metric: str = 'acc',  # which metric to use when choosing threshold value
                 num_threshold_discretizations: int = 200 # how many points to check between min/max y_hat output
                 ):
        super().__init__()
        self.save_hyperparameters()
        seed_everything(seed, workers=True)

        self.parameters()
        self.layer_1 = GCNLayer(num_one_hot+1 if num_one_hot > 0 else 1, channels[0])
        self.layer_2 = GCNLayer(channels[0], channels[1])

        # self.register_buffer('real_data_prior', torch.zeros(1, 68, 68), persistent=True)
        self.register_buffer('threshold', torch.tensor([-1.0]), persistent=True)
        self.test_threshold = None  # placeholder, will save in checkpoint and load when needed

        # private variables to be used for printing/logging
        self.epoch_val_losses = []
        self.list_of_metrics = []

        self.subnetwork_masks = None
        self.checkpoint_loaded = False  # are we being loaded from a checkpoint?

        self.min_output = torch.tensor([-1])
        self.max_output = torch.tensor([1])

    def forward(self, batch) -> Any:
        return self.shared_step(batch=batch)

    def shared_step(self, batch):
        # fcs, adjs, subject_ids, scan_dirs, tasks = batch
        fcs, adjs = batch[:2]
        batch_size, N = fcs.shape[:-1]

        # use contin degrees as node feature
        # [contin_degrees; one_hots]
        node_features = fcs.sum(dim=1).view(batch_size, N, 1)
        node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)

        # input features are the continuos degrees of fcs.
        # A_ = A + I
        # D_ = diag(A_ 1)
        # D_^{-1/2} A_ D_^{-1/2}
        fcs += torch.eye(N, device=self.device).expand(batch_size, N, N)
        fcs = graph_learning.misc.utils.symmetric_adj_normalize(fcs, detach_norm=True)

        if self.hparams.num_one_hot > 0:
            # append on one hot node features for more expressiveness
            ones = torch.ones(batch_size, N, self.hparams.num_one_hot)
            node_features = torch.cat([node_features, ones], dim=2)

        z = torch.nn.functional.leaky_relu(self.layer_1(node_features, fcs), negative_slope=.2) # \in [batch_size, chennels[0]]
        #z = torch.nn.functional.relu(self.layer_1(node_features, fcs)) # \in [batch_size, chennels[0]]
        z = self.layer_2(z, fcs) # \in [batch_size, channels[1]]

        # decode with inner product
        y_hat = z.bmm(z.transpose(-1, -2))
        # **SIGMOID APPLIED IN LOSS IF WE ARE DOING BCE**??
        if self.hparams.which_loss == 'BCE':
            y_hat = torch.sigmoid(y_hat)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch[0:2]

        y_hat = self.shared_step(batch)
        self.max_output = y_hat.max() if not torch.isnan(y_hat.max()) else torch.tensor([3])
        self.min_output = y_hat.min() if not torch.isnan(y_hat.min()) else torch.tensor([-3])

        # only consider off diagonal elements in loss (dont punish diagonal) and computation of metrics
        y_hat, y = adj2vec(y_hat), adj2vec(y)
        loss = self.loss(y=y, y_hat=y_hat)
        metrics = compute_metrics(y_hat=y_hat, y=y, threshold=self.threshold, self_loops=False)
        return {'loss': loss, 'metrics': metrics, 'batch_size': len(x)}

    def loss(self, y, y_hat):
        assert self.hparams.which_loss in ['hinge', 'mse', 'se', 'ae', 'BCE']

        if self.hparams.which_loss == 'BCE':
            return bce_loss(input=y_hat, target=(y>0) + 0.0)
        elif self.hparams.which_loss in ['se', 'mse']:
            return torch.nn.functional.mse_loss(input=y_hat, target=y, reduction='mean')
        elif self.hparams.which_loss == 'hinge':
            return graph_learning.misc.metrics.hinge_loss(y=y, y_hat=y_hat, margin=.25, slope=1, per_edge=False).mean()
        elif self.hparams.which_loss == 'ae':
            return l1_loss(input=y_hat, target=y)

    def training_epoch_end(self, train_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        # self.log("epoch_avg_train_loss", avg_loss, logger=False, prog_bar=True)
        self.log(name=f'train/loss_epoch', value=avg_loss, on_step=False, on_epoch=True)

        means, stdes = self.aggregate_step_outputs(outputs=train_step_outputs)
        self.log_metrics(means, stdes, stage='train')
        return

    def on_validation_start(self) -> None:
        # find best threshold only once before every validation epoch
        # use training set to optimize threshold during training
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=100) #self.hparams.num_threshold_discretizations)
        self.threshold[0] = self.find_threshold(dl=self.trainer.datamodule.train_dataloader(),
                                                threshold_test_points=test_points,
                                                metric2chooseThresh=self.hparams.threshold_metric)
        self.log('threshold', self.threshold, prog_bar=True, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self.shared_step(batch)

        # only consider off diagonal elements in loss (dont punish diagonal) and computation of metrics
        y_hat, y = adj2vec(y_hat), adj2vec(y)
        loss = self.loss(y=y, y_hat=y_hat)
        metrics = compute_metrics(y_hat=y_hat, y=y, threshold=self.threshold, self_loops=False)
        return {'loss': loss, 'metrics': metrics, 'batch_size': len(x)}

    def validation_epoch_end(self, val_step_outputs):
        stage = 'val'
        means, stdes = self.aggregate_step_outputs(outputs=val_step_outputs)

        # save running list of metrics as we go
        self.list_of_metrics.append({'means': means, 'stdes': stdes, 'epoch': self.current_epoch})
        self.log_metrics(means, stdes, stage=stage)

        metrics_in_progress_bar = ['se', 'error', 'f1', 'mcc', 'ae']  # , 'mcc', 'f1']#'ae']
        prog_bar_metrics_dict = {}
        for metric_name in metrics_in_progress_bar:
            prog_bar_metrics_dict[metric_name] = 100 * means[metric_name] if metric_name in ['acc', 'error', 'mcc', 'f1'] else means[metric_name]
            # prog_bar_metrics_dict[metric_name] = 100*means[metric_name] if metric_name not in ['nse', '10_log_nse', 'se', 'se_per_edge', 'ae', 'hinge' 'se_per_edge'] else means[metric_name]
        self.log_dict(prog_bar_metrics_dict, logger=False, prog_bar=True)

        # to able to see/log lr, need to do this
        current_lr = self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        self.log("lr", round(current_lr, 10), logger=True, prog_bar=True)

        # print summary of training results on full network: green good, red bad
        print(f'\nValidation Performance using loss: BCE')
        logged_metrics = ['val/error', 'val/ae',  'val/se', 'val/se_per_edge']  # , 'val/10log_nse']

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
        print(f"\n\tsaving checkpoint...")#: saving threshold found {checkpoint['test_threshold']:.3f} using validation set during training")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.list_of_metrics = checkpoint['list_of_metrics']
        self.checkpoint_loaded = True  # ensures we dont do model setup procedure again
        print(f"\nLoading checkpoint...") #threshold found using validaiton set during training: {self.test_threshold:.5f} which achieved {self.list_of_metrics[-1]['full']['mean']['error'] * 100:.4f}% error")

    def test_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self.shared_step(batch)

        # only consider off diagonal elements in loss (dont punish diagonal) and computation of metrics
        y_hat, y = adj2vec(y_hat), adj2vec(y)
        loss = self.loss(y=y, y_hat=y_hat)
        metrics = compute_metrics(y_hat=y_hat, y=y, threshold=self.threshold, self_loops=False)
        return {'loss': loss, 'metrics': metrics, 'batch_size': len(x)}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        means, stdes = self.aggregate_step_outputs(outputs=outputs)
        self.log_metrics(means, stdes, stage='test')
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    ### HELPER METHODS ###
    @torch.no_grad()
    def find_threshold(self, dl, threshold_test_points, metric2chooseThresh):
        # use data (train or val) set to optimize threshold during training
        ys, y_hats = [], []
        for i, batch in enumerate(iter(dl)):  # loop so dont run out of memory
            batch[0] = batch[0].to(self.device)  # move fcs/scs to GPU (ligthning doesnt do this for us)
            batch[1] = batch[1].to(self.device)
            ys.append(batch[1])
            y_hats.append(self.shared_step(batch))
        y, y_hat = torch.cat(ys, dim=0), torch.cat(y_hats, dim=0)
        y = adj2vec(y) if y.ndim == 3 else y
        y_hat = adj2vec(y_hat) if y_hat.ndim == 3 else y_hat

        # loop over candidate thresholds, see which one optimizes threshold_metric (acc, mcc, se, etc)
        # over FULL network
        return best_threshold_by_metric(thresholds=threshold_test_points,
                                        y=y, y_hat=y_hat, metric=metric2chooseThresh,
                                        device=self.device)

    @torch.no_grad()
    def aggregate_step_outputs(self, outputs):
        # aggregate all outputs from step batches
        # total_epoch_samples = torch.stack([torch.tensor(x['batch_size']) for x in outputs]).sum()
        all_sample_metrics = {m: [] for m in outputs[0]['metrics'].keys()}
        for output in outputs:
            for metric_name, metric_values in output['metrics'].items():
                all_sample_metrics[metric_name].append(metric_values)

        # combine list of tensors into one large tensor

        for metric_name, metric_values in all_sample_metrics.items():
            if 'glad' in metric_name:
                # will throw error bc cant concat single dim tensors
                all_sample_metrics[metric_name] = torch.tensor(metric_values)
            else:
                all_sample_metrics[metric_name] = torch.cat(metric_values)

        # compute mean and standard error of each
        means, stdes = {}, {}
        for metric_name, metric_values in all_sample_metrics.items():
            means[metric_name] = torch.mean(metric_values)
            stdes[metric_name] = torch.std(metric_values) / math.sqrt(len(metric_values))

        return means, stdes

    def best_metrics(self, sort_metric, top_k=1, maximize=True):
        sorted_list_of_metrics = sorted(self.list_of_metrics, key=lambda e: e['means'][sort_metric], reverse=maximize)
        return sorted_list_of_metrics[:top_k]

    def progress_bar_update(self, outputs):
        metrics_in_progress_bar = ['error', 'mcc', 'se', 'nse', 'ae']
        prog_bar_metrics_dict = {}
        for metric in metrics_in_progress_bar:
            value = 100 * outputs[self.hparams.real_network]['mean'][metric] if metric not in ['se', 'ae', 'hinge'] else \
                outputs[self.hparams.real_network]['mean'][metric]
            prog_bar_metrics_dict[metric] = value
        self.log_dict(prog_bar_metrics_dict, logger=False, prog_bar=True)

    def log_metrics(self, means, stdes, stage):

        for metric_name in means.keys():
            name = f'{stage}/{metric_name}'
            self.log(name=name + '/' + 'mean', value=means[metric_name])
            self.log(name=name + '/' + 'stde', value=stdes[metric_name])

    def on_after_backward(self) -> None:
        if self.layer_1.projection.weight.grad.abs().max() < 1e-5 or self.layer_2.projection.weight.grad.abs().max() < 1e-5:
            print(f'\nGradients in layers very small!')
            print(f'\tmax grad layer 1: {self.layer_1.projection.weight.grad.max()}, max_grad_layer 2: {self.layer_2.projection.weight.grad.max()}')