import os, sys, wandb, numpy as np, torch
from pathlib import Path

from graph_learning.misc import metrics

file = Path(__file__).resolve()
path2project = str(file.parents[2]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> dpg/

from graph_learning.data.school_networks.school_networks import SchoolNetworks
from graph_learning.misc.metrics import compute_metrics



def run_batch(wandb, dataloader, name):
    batch = next(iter(dataloader))
    x, y = batch[:2]
    metrics = compute_metrics(y_hat=(x>wandb.config.threshold)+0.0, y=y, threshold=0, reduction = torch.mean)
    metrics = {name + '/' + m: v for m, v in metrics.items()}
    return metrics


def train():
    hyperparameter_defaults = dict(
        threshold=0.7,
        num_samples_val=500, num_samples_test=500,
        seed=50)

    with wandb.init(config=hyperparameter_defaults) as run:
        # build coefficients -> will be the same given same rand_seed
        dm = "" # specify which data to use (smooth, diffuse, brain, etc)
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
    seed = 50
    num_vertices = 120
    dm = SchoolNetworks(num_vertices=num_vertices,
                        seed=seed, batch_size=200 if 'max' in os.getcwd() else 64,
                        num_workers=0 if 'max' in os.getcwd() else 4,
                        # train_size=5000, val_size=1000, test_size=1000,
                        train_size=750, val_size=100, test_size=200,
                        x_graphs='sensor_contact', y_graphs='facebook_friend',
                        x_gso='adjacency', y_gso='adjacency', x_min_eig=None, y_min_eig=None
                        )
    dm.setup('fit')
    x, y = dm.train_ds.dataset.tensors
    #x, y = train_ds[0], train_ds[1]
    thresholds = torch.linspace(0.0, x.max(), 150)
    train_threshold = metrics.best_threshold_by_metric(y_hat=x, y=y, thresholds=thresholds, non_neg=True, metric='error')
    train_metrics = compute_metrics(y_hat=x, y=y, threshold=train_threshold, self_loops=False, non_neg=True)

    x_test, y_test = dm.test_ds.dataset.tensors
    test_metrics = compute_metrics(y_hat=x_test, y=y_test, threshold=train_threshold, self_loops=False, non_neg=True)

    print(f"threshold {train_threshold:4f} tuned on train gets error {train_metrics['error'].mean():.5f} on train, error {test_metrics['error'].mean():.5f} on test")
    print(f"test error stde: {test_metrics['error'].std()/np.sqrt(len(test_metrics['error']))}")
