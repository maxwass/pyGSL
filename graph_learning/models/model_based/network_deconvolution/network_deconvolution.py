import os, sys, wandb, numpy as np, torch
from pathlib import Path

file = Path(__file__).resolve()
path2project = str(file.parents[2]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> dpg/

from graph_learning.data.school_networks.school_networks import SchoolNetworks
from graph_learning.misc.metrics import compute_metrics, best_threshold_by_metric


def network_deconvolution(x, alpha=1.0):
    assert x.ndim == 3
    batch_size, N, N1 = x.shape
    assert N == N1

    vals, vecs = torch.linalg.eigh(x)
    #assert torch.max(vals) <= 1.0, f'network deconv: max eigenvalue should be normalized'

    nd_vals = vals / (1 + alpha*vals)  # network deco
    nd = torch.matmul(torch.matmul(vecs, torch.diag_embed(nd_vals)), torch.transpose(vecs, 1, 2) )
    return nd


if __name__ == "__main__":
    # train()
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
    train_errors, test_errors_mean, test_errors_stde = [], [], []
    num_thresholds = 100
    num_alphas = 5
    alphas = torch.logspace(torch.log10(torch.tensor([.0001])).item(), torch.log10(torch.tensor([.5])).item(), base=10, steps=num_alphas)
    for alpha in alphas:
        y_hat = network_deconvolution(x, alpha=alpha)
        thresholds = torch.linspace(0.0, y_hat.max(), num_thresholds)
        train_threshold = best_threshold_by_metric(y_hat=y_hat, y=y, thresholds=thresholds, non_neg=True, metric='error')
        train_metrics = compute_metrics(y_hat=y_hat, y=y, threshold=train_threshold, self_loops=False, non_neg=True)

        x_test, y_test = dm.test_ds.dataset.tensors
        test_metrics = compute_metrics(y_hat=x_test, y=y_test, threshold=train_threshold, self_loops=False, non_neg=True)
        train_errors.append(train_metrics['error'].mean().item())
        test_errors_mean.append(test_metrics['error'].mean().item())
        test_errors_stde.append(test_metrics['error'].std().item()/np.sqrt(len(test_metrics['error'])))
        print(f"alpha {alpha:.5f}: train {train_metrics['error'].mean():.5f}, test {test_metrics['error'].mean():.5f}")
        #print(f"threshold {train_threshold:4f} tuned on train gets error {train_metrics['error'].mean():.5f} on train, error {test_metrics['error'].mean():.5f} on test")

    train_errors = np.array(train_errors)
    best_train_error_idx = np.argmin(train_errors)
    generalization_test_error = test_errors[best_train_error_idx]
    print(f'best train error occured at alpha {alphas[best_train_error_idx]:.5f}')
    print(f'\ttrain error  {train_errors[best_train_error_idx]:.5f}')
    print(f'\ttest error mean  {test_errors_mean[best_train_error_idx]:.5f}')
    print(f'\ttest error stde  {test_errors_stde[best_train_error_idx]:.5f}')

