import math, sys, numpy as np, torch, pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import seed_everything
from typing import Any, Dict, List, Optional, Type

import graph_learning.misc.utils
from graph_learning.misc.utils import adj2vec
from graph_learning.misc.metrics import best_threshold_by_metric, hinge_loss

from graph_learning.misc.metrics import compute_metrics

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''

        super(MLP, self).__init__()

        self.num_layers = num_layers

        # Multi-layer model
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #for l in self.linears:
        #    torch.nn.init.xavier_normal_(l.weight, gain=1)

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h_t = self.linears[layer](h).transpose(-1, -2)
            h = F.relu(self.batch_norms[layer](h_t)).transpose(-1, -2)
        return self.linears[self.num_layers - 1](h)


class GINLayer(nn.Module):
    def __init__(self, c_in, c_out, num_layers, hidden_dim):
        super().__init__()
        self.mlp = MLP(num_layers=num_layers, input_dim=c_in, hidden_dim=hidden_dim, output_dim=c_out)
        self.batch_norm = nn.BatchNorm1d(c_out)

    def forward(self, node_feats, adj_matrix, eps, include_relu = True):
        """
        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
            eps: Tensor scalar to scale the self feature
        """
        # Node representation update rule: node v at layer k, w_{u,v} is edge weight between u and v
        # h_v^(k) = MLP[ (1+eps) h_v^(k-1) + \sum_neighbors_u w_{u,v} * h_u^(k-1) ]
        # for GIN-0 this reduces to
        # h_v^(k) = MLP[ h_v^(k-1) + \sum_neighbors_u w_{u,v} * h_u^(k-1) ]

        # Let H be a matrix in [batch_size, num_nodes, num_feats]
        # Let A be a weighted adjacency W/O self loops: [batch_size, num_nodes, num_nodes]
        # Update Rule: H^(k) = MLP[ (1+eps) H^{k-1} + A * H^{k-1} ]

        # POOLING
        # mean/sum aggregation of neighbors (depending on normalization of A)
        #assert torch.allclose(torch.diag(adj_matrix), torch.zeros([1])), f'diagonal of adjacency matrices must be zero! Dont include self-loops.'
        mean_agg = torch.bmm(adj_matrix, node_feats) # A * h \in [batch_size, num_nodes, c_in]
        self_agg = (1+eps) * node_feats # \in [batch_size, num_nodes, c_in]
        pooled = mean_agg + self_agg
        # TRANSFORM
        out = self.mlp(pooled) # \in [batch_size, num_nodes, c_out]

        # this just repeats the operation in MLP...simply suck this into the MLP??
        p = self.batch_norm(out.transpose(-1, -2)).transpose(-1, -2)
        return F.relu(p) if include_relu else p


bce_loss = torch.nn.BCELoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')
class gin(pl.LightningModule):
    def __init__(self,
                 # architecture
                 num_layers: int = 5,
                 num_mlp_layers: int = 2,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
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

        input_dim = 1 if num_one_hot == 0 else 1 + num_one_hot
        layers = []
        for layer in range(num_layers-1):
            layers.append(GINLayer(c_in=input_dim if layer == 0 else hidden_dim,
                                   c_out=hidden_dim if layer != num_layers-2 else output_dim,
                                   num_layers=num_mlp_layers,
                                   hidden_dim=hidden_dim))

        self.GIN_layers = nn.ModuleList(layers)

        # Needed??
        # Linear function that maps the hidden representation at differemt layers into a prediction score
        #linears_prediction = torch.nn.ModuleList()
        #for layer in range(num_layers):
        #    linears_prediction.append(nn.Linear(input_dim if layer == 0 else hidden_dim, output_dim))
        #self.linears_prediction = nn.ModuleList(linears_prediction)


        ####

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
        a_o, a_l = batch[:2]
        batch_size, N = a_o.shape[:-1]

        # use contin degrees as node feature
        # [contin_degrees; one_hots]
        node_features = a_o.sum(dim=1).view(batch_size, N, 1)
        node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)

        # A_ = A + I
        # D_ = diag(A_ 1)
        # D_^{-1/2} A_ D_^{-1/2}
        # REMOVE DIAGONAL FROM GRAPH -> TREAT SELF-EDGE DIFFERENTTLY
        #a_o = a_o - torch.diag_embed(torch.diag(a_o), device=self.device) # CHECK DEVICE!!
        zd = (torch.ones((N, N), device=a_o.device) - torch.eye(N, device=a_o.device)).expand(a_o.shape)
        a_o = a_o * zd
        a_o = graph_learning.misc.utils.symmetric_adj_normalize(a_o, detach_norm=True)

        if self.hparams.num_one_hot > 0:
            # append on one hot node features for more expressiveness
            ones = torch.ones(batch_size, N, self.hparams.num_one_hot)
            node_features = torch.cat([node_features, ones], dim=2)

        h = node_features
        for layer in range(self.hparams.num_layers-1):
            # when not to include relu: when doing BCE on the final layer -> need non-neg entries for inner product to
            # output negative things and thus sigmoid to work effectively around 0
            exclude_relu = (layer == self.hparams.num_layers - 2) and self.hparams.which_loss == 'BCE'
            h = self.GIN_layers[layer](node_feats=h, adj_matrix=a_o, eps=0.0, include_relu=not exclude_relu)

        # h \in [batch_size, num_nodes, output_dim]
        # decode with inner product
        y_hat = h.bmm(h.transpose(-1, -2))
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
        test_points = torch.linspace(start=self.min_output.item(), end=self.max_output.item(), steps=200) #self.hparams.num_threshold_discretizations)
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

    #def configure_optimizers(self):
    #    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #    return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]

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