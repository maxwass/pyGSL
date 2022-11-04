import math, sys, torch
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

file = Path(__file__).resolve()
path2project = str(file.parents[3]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/


from graph_learning.misc.metrics import regression_metrics, symmetric_classification_metrics


class iterative():
    """Requires methods: shared_step,  """
    """Requires attributes: max_output, min_output, list_of_metrics, threshold, testing_threshold, """
    def __init__(self):
        super().__init__()

    def _suboptimality_gap(self, optim_variables):
        # compute suboptimality gap
        return self._primal_value(optim_variables) - self._dual_value(optim_variables)

    @staticmethod
    def _compute_metrics(y_hat, y):

        regress_metrics = regression_metrics(y_hat=y_hat, y=y, self_loops=False)

        # requires a threshold
        #classification_metrics = symmetric_classification_metrics(y_hat=y_hat, y=y, threshold=threshold)

        # compute mean and standard error of each
        means, stdes = {}, {}
        for metric_name, metric_values in regress_metrics.items():
            means[metric_name + '/' + 'mean'] = torch.mean(metric_values)
            stdes[metric_name + '/' + 'stde'] = torch.std(metric_values) / math.sqrt(len(metric_values))
        return {**means, **stdes}

    @staticmethod
    def _stopping_conditions(d: Dict,  eps: float):
        # d is dict of l2-norm of change in each optimization variable
        # ex) {'w': [.23, .45, ..., .54], ... }
        # compute ~change~ on primal and dual:\|w[i] - w[i-1]\|_2  and \|v[i] - v[i-1]\|_2 -> stopping conditions!

        stop_conditions = {}
        stop_conditions_bool = []
        for name, l2_norm_diffs in d.items():
            stop_conditions['2norm_diff_' + 'ave_' + name] = l2_norm_diffs.mean()
            if not 'mps' in l2_norm_diffs.device.type:
                # median operation not yet implemented for mps
                stop_conditions['2norm_diff_' + 'median_' + name] = l2_norm_diffs.median()
            stop_conditions['2norm_diff_' + 'max_' + name] = l2_norm_diffs.max()
            stop_conditions_bool.append(l2_norm_diffs.max() < eps)

        # the largest change must be smaller than epsilon
        stop = all(stop_conditions_bool)

        return stop, {'stop_conditions/' + k: v for (k, v) in stop_conditions.items()}


if __name__ == "__main__":
    print('iterative base mixin main loop')

