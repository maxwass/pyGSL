"""
    Spectral Template is a Graph Structure Learning model introduced by Segarra et al
    in "Network Topology Inference from Spectral Templates", https://arxiv.org/abs/1608.03008

    This relies on CVXPY's solver, https://www.cvxpy.org.

    Note we have found the CVXPY python package has not played nicely with other packages. We reccomend creating
    a separate conda environment. We advise users to make a seperate conda environment with only the packages
    needed in this file.


    ----
    CVXPY requires MOSEK for fast execution. MOSEK must be downloaded manually. MOSEK provides free license for
    academic purposes. Follow the instructions here: https://www.mosek.com/products/academic-licenses/


"""

import sys, os, numpy as np, time, torch, cvxpy as cp, wandb, mosek
from cvxpy import error
from itertools import compress
from torch.linalg import eigh
from pathlib import Path
file = Path(__file__).resolve()
path2project = str(file.parents[2]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/

from graph_learning.misc.utils import edge_density, sample_spherical
from baselines.baseline_utils import zero_diagonals, min_mse_regressions, find_mse, make_synthetic_datamodule, make_pseudo_synthetic_datamodule, wandb_setup, find_best_performances
np.set_printoptions(precision=4)

wandb_setup(offline_mode='max' in os.getcwd())


########### cvx/spec temp helper functions #############
# getter function to access cvxpy problem parameters and attributes
def problem_dicts(problem):
    param_dict = {x.name(): x for x in problem.parameters()}
    variable_dict = {x.name(): x for x in problem.variables()}
    return param_dict, variable_dict


# for iterative_reweighted procedure
def compute_weights(S_prev, tau, delta):
    ones_mat = np.ones_like(S_prev)
    weights_val = np.divide(tau * ones_mat, np.abs(S_prev) + delta * ones_mat)
    return weights_val


########### Spectral Templates Meat #############
# constructors for the two cvxpy problems. Inner spectral templates, then
# an iterative re-weighting to clean up small weights.
def spectral_template_problem(N, eps=None, spec_temps_in=None):
    # Define Variables and Parameters
    S_hat = cp.Variable((N, N), name='S_hat', symmetric=True)
    S = cp.Variable((N, N), name='S', symmetric=True)
    lam = cp.Variable(N)
    epsilon = cp.Parameter(nonneg=True, name='epsilon', value=eps)# if (eps != None) else None)
    spec_temps = cp.Parameter((N, N), 'eigenvectors', value=spec_temps_in)# if (spec_temps_in != None) else None)

    # Define objective and constraints
    objective = cp.Minimize(cp.sum(cp.abs(S)))  # cp.Minimize(cp.norm(S.flatten(), 1)) # CHECK CORRECTNESS: standard way to do sum of abs vals?
    constraints = [S_hat == spec_temps @ cp.diag(lam) @ spec_temps.T,
                   S >= 0,
                   cp.abs(cp.diag(S)) <= 1e-6,
                   S @ np.ones(N) >= 1,
                   cp.norm(S - S_hat, 'fro') <= epsilon]

    # Solve
    prob = cp.Problem(objective=objective, constraints=constraints)
    #assert prob.is_dcp(dpp=True), f'problem must comply with DPP rules for fast resolving.'
    return prob


def iterative_reweighted_problem(N, eps, spec_temps_in=None):
    # Define Variables and Parameters
    S_hat = cp.Variable((N, N), name='S_hat', symmetric=True) #this differs from matlab code. Dis he make mistake?
    S = cp.Variable((N, N), name='S', symmetric=True)
    lam = cp.Variable(N)
    epsilon = cp.Parameter(name='epsilon', nonneg=True, value=eps)
    spec_temps = cp.Parameter((N, N), name='eigenvectors', value=spec_temps_in)

    weights = cp.Parameter((N, N), name='weights', nonneg=True)

    # Define objective and constraints
    objective = cp.Minimize(cp.sum(cp.multiply(weights, S))) # elementwise multiply by weights
    constraints = [S_hat == spec_temps @ cp.diag(lam) @ spec_temps.T,
                   S >= 0,
                   cp.abs(cp.diag(S)) <= 1e-6,
                   S @ np.ones(N) >= 1,
                   cp.norm(S - S_hat, 'fro') <= epsilon]

    # Solve
    prob = cp.Problem(objective=objective, constraints=constraints)
    #assert prob.is_dcp(dpp=True), f'problem must comply with DPP rules for fast resolving.'
    return prob


# run spectral_templates + iterative rewighting on a SINGLE sample
def spectral_templates(emp_cov: np.ndarray, emp_cov_eigenvectors: np.ndarray, epsilon_range=(0, 2),
                       binary_search_iters: int=5,
                       tau=1, delta=.001, return_on_failed_iter_rew: bool=True,
                       num_iter_reweight_refinements:int = 3):
    N = emp_cov.shape[-1]
    st_prob = spectral_template_problem(N, spec_temps_in=emp_cov_eigenvectors)
    st_param_dict, st_variable_dict = problem_dicts(st_prob)

    #############
    # Perform Binary Search on epsilon value. Resolve convex Spectral Templates problem for each epsilon.
    # Find smallest epsilon which allows a solution.
    epsilon_low, epsilon_high = epsilon_range
    # S_prev = solution to spectral temaples problem with smallest working epsilon
    smallest_working_epsilon, S_prev = None, None
    for i in range(binary_search_iters):
        epsilon = (epsilon_low + epsilon_high)/2
        st_param_dict['epsilon'].value = epsilon
        try:
            st_prob.solve(solver='MOSEK', warm_start=True, verbose=False)
            if st_prob.status == 'optimal':
                worked = True
                print(f'\tSpecTemp: {i}th iteration took: {st_prob.solver_stats.solve_time:.4f} s')#,  {st_prob.solver_stats.num_iters} iterations')
            else:
                # infeasible, unbounded, etc
                worked = False
                print(f'\t{i}th binary search iteration failed: {st_prob.status}')
        except error.SolverError as e:
            worked = False
            print(f'\t{i}th binary search iteration threw CVX exception: {e}')
        except Exception as e:
            worked = False
            print(f'\t{i}th binary search iteration threw OTHER exception: {e}')

        if worked:
            # worked, try smaller epsilon => smaller radius of Euclidean ball around S_hat
            epsilon_high = epsilon
            smallest_working_epsilon = epsilon
            S_prev = st_variable_dict['S'].value
        else:
            # didn't work, try larger epsilon => larger radius of Euclidean ball around S_hat
            epsilon_low = epsilon

    if S_prev is None:
        raise ValueError(f'\tNone of the epsilons in {epsilon_range} worked')

    #############
    # now apply iterative reweighting scheme a few times to clean up small edge weights.
    iter_rewt_prob = iterative_reweighted_problem(N=N, eps=st_param_dict['epsilon'].value, spec_temps_in=emp_cov_eigenvectors)
    iter_rewt_param_dict, iter_rewt_variable_dict = problem_dicts(iter_rewt_prob)

    worked = False
    for i in range(num_iter_reweight_refinements):
        iter_rewt_param_dict['weights'].value = compute_weights(S_prev=S_prev, tau=tau, delta=delta)
        # include try/except here for when solver fails. Better printing.
        try:
            iter_rewt_prob.solve(solver='MOSEK', warm_start=True, verbose=False)
            if iter_rewt_prob.status == 'optimal':
                worked = True
                print(f'\tIter Refine: {i}th iteration took: {iter_rewt_prob.solver_stats.solve_time:.4f} s')#,  {iter_rewt_prob.solver_stats.num_iters} iterations')
            else:
                # infeasible, unbounded, etc
                worked = False
                print(f'\t{i}th Iterative Reweighting iteration failed: {iter_rewt_prob.status}')

        except error.SolverError as e:
            worked = False
            print(f'\t{i}th Iterative Reweighting iteration threw CVX exception: {e}')
        except Exception as e:
            worked = False
            print(f'\t{i}th Iterative Reweighting iteration threw OTHER exception: {e}')

        if worked:
            S_prev = iter_rewt_variable_dict['S'].value
        elif return_on_failed_iter_rew:
            if i>0:
                print(f'\t\tReturning {i - 1} Iterative Reweighting soln')
            else:
                print(f'\t\tReturning Spectral Templates solution with NO Iterative Reweighting applied')

            return S_prev, smallest_working_epsilon, (i+1)

        else:
            raise ValueError(f'Iterative Reweighting Failed: '
                             f'To return last valid solution, set return_on_failed_iter_rew <- True')

    return S_prev, smallest_working_epsilon, num_iter_reweight_refinements


########### wrappers to run on batches of samples #############3
# run the spectral_templates + iterative rewighting over a batch of samples
def spectral_templates_batch(emp_cov_batch: np.ndarray, emp_cov_eigvecs_batch: np.ndarray, allowable_failures: int):
    if emp_cov_eigvecs_batch is None:
        emp_cov_eigvecs_batch = eigh(emp_cov_batch)[1]

    y_hats, epsilons = [], []
    successful_runs, num_iter_reweighting_refinements_completed = [], []
    for i in range(len(emp_cov_batch)):
        try:
            print(f'##########')
            print(f'{i}th sample - start binary search')
            start = time.time()
            y_hat, epsilon, a = spectral_templates(emp_cov=emp_cov_batch[i], emp_cov_eigenvectors=emp_cov_eigvecs_batch[i])
            y_hats.append(y_hat); epsilons.append(epsilon); successful_runs.append(True); num_iter_reweighting_refinements_completed.append(a)
            print(f'{i}th sample complete: {time.time() - start:.3f}s')
        except Exception as e:
            print(f'{i}th sample - exception: {e}')
            y_hats.append(None); epsilons.append(None); successful_runs.append(False); num_iter_reweighting_refinements_completed.append(None)
        num_failures = sum([f is None for f in y_hats])
        if num_failures > allowable_failures:
            raise ValueError(f'this batch has failed {num_failures} times. More than the {allowable_failures} allowable failures.')

    print(f'batch completed with {np.sum(successful_runs)}/{len(emp_cov_batch)} completing succesfully ')
    return y_hats, epsilons, successful_runs, num_iter_reweighting_refinements_completed


def spec_temp_batch(x, y, eps, allowable_failure_rate=.5):
    emp_cov_eigvals, emp_cov_eigvecs = eigh(x)
    y_hats, epsilons, successful_indices, num_iter_reweighting_refinements_completed = \
        spectral_templates_batch(emp_cov_batch=x, emp_cov_eigvecs_batch=emp_cov_eigvecs.numpy(),
                                 allowable_failures=int(allowable_failure_rate*len(x)))

    # remove runs which did not complete successfully
    y_hats, epsilons, num_iter_reweighting_refinements_completed = \
        list(compress(y_hats, successful_indices)), list(compress(epsilons, successful_indices)), \
        list(compress(num_iter_reweighting_refinements_completed, successful_indices))
    y_hat = np.stack(y_hats, axis=0)
    y = y[successful_indices, :, :]

    y_hat = zero_diagonals(torch.tensor(y_hat))
    return y_hat
    """
    metrics, threshold, regressions = find_best_performances(torch.tensor(y), torch.tensor(y_hat), threshold, regressions)

    metrics['failure_rate'] = 1 - len(successful_indices) / len(x)

    print(f'\n### Prediction Stats: using eps {eps:.5f}, threshold {threshold} ###')
    print(f'\tpred edge weights: max: {np.nanmax(y_hat):.3f}, median {np.nanmedian(y_hat):.3f}, mean {np.nanmean(y_hat):.3f}')
    print(f'\tave sparsity of true graphs: {edge_density(y).mean():.3f}')
    print(f'\tave sparsity of (i) raw pred: {edge_density(y_hat).mean():.3f}, (ii) pred > {threshold:.7f}:  {edge_density((y_hat > threshold) + 0.0).mean():.3f}')
    print(f'\tepsilons: mean: {np.nanmean(epsilons):.3f}, std: {np.nanstd(epsilons):.3f} ... epsilons {epsilons}')
    print(f"\tfailure rate: {100*metrics['failure_rate']:.4f}%")
    print(f'### Performance ###')
    print(f"\terrors {metrics['error']*100:.5f}%, using found best threshold: {threshold:.7f}")
    print(f"\tmses:  raw: {metrics['raw_mse']:.5f}, ols: {metrics['ols_mse']:.5f}, scaling_mse: {metrics['ols_no_intercept_mse']:.5f}")
    print('\t# iterative refines', num_iter_reweighting_refinements_completed)
    return metrics, threshold, regressions
    """


def run_batch(wandb, dataloader, threshold=None, regressions=None):
    batch = next(iter(dataloader))
    eps, allowable_failure_rate = wandb.config.eps, wandb.config.allowable_failure_rate
    x, y, _, _, _ = batch

    emp_cov_eigvals, emp_cov_eigvecs = eigh(x)
    y_hats, epsilons, successful_indices, num_iter_reweighting_refinements_completed = \
        spectral_templates_batch(emp_cov_batch=x, emp_cov_eigvecs_batch=emp_cov_eigvecs.numpy(),
                                 allowable_failures=int(allowable_failure_rate*len(x)))

    # remove runs which did not complete successfully
    y_hats, epsilons, num_iter_reweighting_refinements_completed = \
        list(compress(y_hats, successful_indices)), list(compress(epsilons, successful_indices)), \
        list(compress(num_iter_reweighting_refinements_completed, successful_indices))
    y_hat = np.stack(y_hats, axis=0)
    y = y[successful_indices, :, :]

    y_hat = zero_diagonals(torch.tensor(y_hat))
    metrics, threshold, regressions = find_best_performances(torch.tensor(y), torch.tensor(y_hat), threshold, regressions)

    metrics['failure_rate'] = 1 - len(successful_indices) / len(x)

    print(f'\n### Prediction Stats: using eps {eps:.5f}, threshold {threshold} ###')
    print(f'\tpred edge weights: max: {np.nanmax(y_hat):.3f}, median {np.nanmedian(y_hat):.3f}, mean {np.nanmean(y_hat):.3f}')
    print(f'\tave sparsity of true graphs: {edge_density(y).mean():.3f}')
    print(f'\tave sparsity of (i) raw pred: {edge_density(y_hat).mean():.3f}, (ii) pred > {threshold:.7f}:  {edge_density((y_hat > threshold) + 0.0).mean():.3f}')
    print(f'\tepsilons: mean: {np.nanmean(epsilons):.3f}, std: {np.nanstd(epsilons):.3f} ... epsilons {epsilons}')
    print(f"\tfailure rate: {100*metrics['failure_rate']:.4f}%")
    print(f'### Performance ###')
    print(f"\terrors {metrics['error']*100:.5f}%, using found best threshold: {threshold:.7f}")
    print(f"\tmses:  raw: {metrics['raw_mse']:.5f}, ols: {metrics['ols_mse']:.5f}, scaling_mse: {metrics['ols_no_intercept_mse']:.5f}")
    print('\t# iterative refines', num_iter_reweighting_refinements_completed)
    return metrics, threshold, regressions

# hyperparameter approximate best ranges
# geom         -> {eps: [1e-6, 1e-2], threshold: ~.023 } -> error = 13%
# ER           -> {eps: [1e-6, 1e-2], threshold: ~1e-6 } -> error = 44%
# pref_attach  -> {eps: [1e-6, 1e-2], threshold: .035 } -> error = 30%
# SBM          -> {eps: [1e-6, 1e-3], threshold: .0235} -> error = 22.6%
# SC           -> {eps: [1e-6, 1e-3], threshold: .02563} -> error = 30%

def train():
    hyperparameter_defaults = dict(
        graph_gen='geom',
        coeffs_index=1,
        eps=.005,
        allowable_failure_rate=.05,
        num_vertices=68,
        fc_norm=None,
        sum_stat="sample_cov",
        num_signals=50,
        num_samples_train=0, num_samples_val=1, num_samples_test=5,
        rand_seed=50)
    if 'max' in os.getcwd():
        os.environ["WANDB_MODE"] = "offline"

    with wandb.init(config=hyperparameter_defaults) as run:
        # build coefficients -> will be the same given same rand_seed
        all_coeffs = sample_spherical(npoints=3, ndim=3, rand_seed=wandb.config.rand_seed)
        if wandb.config.graph_gen != 'SC':
            dm = make_synthetic_datamodule(wandb, all_coeffs)
        else:
            dm = make_pseudo_synthetic_datamodule(wandb, all_coeffs)
        dm.setup('fit')

        print(f'\n\nVALIDATION')
        try:
            metrics, threshold, regressions = run_batch(wandb, dataloader=dm.val_dataloader())
            metrics = {'val/' + m: v for m, v in metrics.items()}
            run.log(data=metrics)
            run.log({'threshold': threshold,
                     'ols_coeff': regressions['ols'].coef_[0], 'ols_intercept': regressions['ols'].intercept_[0],
                     'ols-wo-intercept_coeff': regressions['ols_no_intercept'].coef_[0]})
            run.log({'val/convergence': True})
            print(f'val metrics: {metrics}')
            del metrics #so dont reuse by accident
        except Exception as e:
            run.log(data={'val/error': 1, 'val/mse': 1e8, 'convergence': False})
            raise RuntimeError(f'Validation Failed: {e}.')

        print(f'\n\nTEST') #comment out during search
        try:
            metrics, _, _ = run_batch(wandb, dataloader=dm.test_dataloader(), threshold=threshold, regressions=regressions)
            metrics = {'test/' + m: v for m, v in metrics.items()}
            run.log(data=metrics)
            run.log({'test/convergence': True})
            print(f'test metrics: {metrics}')
            del metrics
        except Exception as e:
            raise RuntimeError(f'Test Failed: {e}.')
        if 'colab' in os.getcwd():
            wandb.finish()  # only needed on colab

if __name__ == "__main__":
    train()

