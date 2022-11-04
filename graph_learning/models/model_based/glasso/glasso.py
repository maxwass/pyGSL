import os, sys, wandb, numpy as np, torch
from pathlib import Path

import graph_learning.misc.utils
from math import sqrt

file = Path(__file__).resolve()
path2project = str(file.parents[2]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> dpg/

from graph_learning.data.network_diffusion.diffused_signals import DiffusionDataset
from graph_learning.data.smooth_signals.smooth_signals import Smooth
from graph_learning.misc.metrics import best_threshold_by_metric, compute_metrics
from graph_learning.misc import metrics
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, graphical_lasso
from graph_learning.misc.utils import adj2vec


"""
    To the authors knowledge, efficient batched GPU implementations of GLASSO are not yet available in python.
    Thus rely on sklearn's single sample implementation. Wrap to run on batch.
"""


def glasso_batch(x, y, alpha, mode='lars', tol=1e-6, max_iter=100):
    y_hat = torch.zeros_like(y)
    for i in range(len(x)):
        x_np, y_hat_np = graphical_lasso(emp_cov=x[i].numpy(), alpha=alpha, mode=mode, tol=tol, max_iter=max_iter)
        y_hat[i] = torch.tensor(y_hat_np)
    return y_hat


def run_batch(wandb, dataloader, name, test=False):
    batch = next(iter(dataloader))
    x, y = batch[:2]
    y_hat = glasso_batch(x, y, alpha=wandb.alpha, mode=wandb.mode, tol=wandb.tol, max_iter=wandb.max_iter)
    if not test:
        # if we're not testing, find best threshold by discretizing
        thresholds = torch.linspace(start=0, end=y_hat.abs().max(), steps=100)
        best_threshold = best_threshold_by_metric(y_hat.abs(), y.abs(), thresholds=thresholds)
    threshold = best_threshold if not test else wandb.config.threshold

    metrics_ = metrics.compute_metrics_glad(y_hat=y_hat, y=y, threshold=threshold, reduction=torch.mean)
    metrics_ = {name + '/' + m: v for m, v in metrics_.items()}
    return metrics


def train():
    hyperparameter_defaults = dict(
        alpha=0.1, #0.03, #075
        mode='lars',
        tol=1e-6,
        max_iter=100,
        threshold=0.7,
        num_samples_val=500, num_samples_test=500,
        seed=50)

    with wandb.init(config=hyperparameter_defaults) as run:
        # build coefficients -> will be the same given same rand_seed
        dm = Smooth(val_size=wandb.config.num_samples_val, test_size=wandb.config.num_samples_test,
                                           data_dir="/Users/maxwasserman/Desktop/dpg/data/data_2000sample_50node_RG.npz",
                                           batch_size=wandb.config.num_samples_val,
                                           seed=wandb.config.seed)
        dm.setup('fit')
        # validation
        metrics_val = run_batch(wandb, dataloader=dm.val_dataloader(), name='val')
        run.log(data=metrics_val)

        # test - do this now so don't have to re run later
        metrics_test = run_batch(wandb, dataloader=dm.test_dataloader(), name='test')
        run.log(data=metrics_test)

        if 'colab' in os.getcwd():
            wandb.finish()  # only needed on colab


if __name__ == '__main__':
    #train()
    """
    from data import PrecisionCovarianceDataset
    train_size, val_size, test_size, batch_size = 3, 5, 5, 5
    dm = PrecisionCovarianceDataset(train_size=train_size, val_size=val_size, test_size=test_size,
                                    batch_size=batch_size,
                                    # u_min=.5, u_max=0.1,
                                    p=0.1, min_density=0.098, max_density=0.102,
                                    u_min=-1.0, u_max=1.0,
                                    min_eig=1.0,
                                    n=100, num_signals=500, num_workers=0)
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    x, y = batch[:2]
    alpha = 0.03#, #075
    y_hat = glasso_batch(x, y, alpha=alpha, mode='lars', tol=1e-6, max_iter=100)

    # compute regression metrics strictly how is it computed in paper (which I think is inccorect):
    # nmse := E || theta_hat - theta ||_F^2 / E || theta ||_F^2
    thresholds = torch.linspace(y_hat.min(), y_hat.max(), 500)
    threshold = best_threshold_by_metric(y_hat=adj2vec(y_hat), y=adj2vec(y), metric='acc', thresholds=thresholds)
    ms = metrics.compute_metrics_glad(y_hat=y_hat, y=y, threshold=threshold)
    print(f"glad nmse: {ms['nmse_glad']:.4f} \nnmse: {ms['nse']} \nse: {ms['se']} \nerror: {ms['error']*100}, \nmcc: {ms['mcc']*100}")
    """

    from graph_learning.data.network_diffusion.diffused_signals import Diffused
    import warnings
    warnings.filterwarnings("ignore")
    train_size = val_size = test_size = batch_size = 3#, 5, 5, 5
    num_coeffs_sample = 3
    all_coeffs = graph_learning.misc.utils.sample_spherical(npoints=num_coeffs_sample, ndim=3, rand_seed=50)
    graph_sampling_params = {'graph_sampling': 'geom', 'num_vertices': 68, 'r': .56, 'dim': 2, 'edge_density_low': .5, 'edge_density_high': .6}
    dm = Diffused(train_size=10, val_size=50, test_size=1, sum_stat='analytic_corr',
                             graph_sampling_params=graph_sampling_params,
                             coeffs=all_coeffs[:, 1], seed=50, sum_stat_norm='max_eig', sum_stat_norm_val='symeig')
    dm.setup('fit')
    # find good alpha
    print(f'HP search for good alpha to optimize error')
    batch = next(iter(dm.train_dataloader()))
    x, y = batch[:2]
    print(f'num_samples: {len(x)}')
    for alpha in np.linspace(5e-3, .0161, num=10): # best for geom -> .01 = 6.8% error
        try:
            y_hat = glasso_batch(x, y, alpha=alpha, mode='cd', max_iter=200, tol=1e-5)
        except Exception as e:
            print(e)
            continue

        threshold = best_threshold_by_metric(y_hat=adj2vec(y_hat.abs()), y=adj2vec(y), metric='acc', thresholds=torch.linspace(0, y_hat.abs().max(), 500))
        ms = metrics.compute_metrics(y_hat=y_hat.abs(), y=y, threshold=threshold, self_loops=False)
        try:
            print(f"alpha = {alpha:.4f}: error: {ms['error'].mean():.3f}, f1: {ms['f1'].mean():.4f}, mcc {ms['mcc'].mean():.4f}")
        except:
            print(f'failed print')

    # test found good alpha
    num_samples = dm.val_size
    print(f'Test good alpha {alpha:.4f} on unseen data: # val samples {num_samples}')
    alpha = .01
    batch = next(iter(dm.val_dataloader()))
    x, y = batch[:2]
    y_hat = glasso_batch(x, y, alpha=alpha, mode='cd', max_iter=200, tol=1e-5)
    thresholds = torch.linspace(0, y_hat.abs().max(), 500)
    threshold = best_threshold_by_metric(y_hat=adj2vec(y_hat.abs()), y=adj2vec(y), metric='acc', thresholds=thresholds)
    ms = metrics.compute_metrics(y_hat=y_hat.abs(), y=y, threshold=threshold, self_loops=False)
    print(f"alpha = {alpha:.4f}: error mean: {ms['error'].mean():.5f}, error stde: {ms['error'].std()/sqrt(len(ms['error'])):.5f},  f1: {ms['f1'].mean():.5f}, mcc {ms['mcc'].mean():.4f}")
