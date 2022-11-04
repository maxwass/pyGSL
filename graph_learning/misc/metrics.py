"""
    Functionality to compute regression and classification metrics for graph prediction.

"""

from __future__ import print_function, division
import numpy as np, torch
from graph_learning.misc.utils import adj2vec
from sklearn import metrics


def perfect_predictions(tp, tn, fp, fn):
    return (tp+tn) == (tp+tn+fp+fn)


def all_incorrect_predictions(tp, tn, fp, fn):
    return (fp+fn) == (tp+tn+fp+fn)


# performance metrics: These functions take two 1D vectors and output scalar metric
def accuracy(tp, tn, total):
    return (tp+tn) / total


def matthew_corr_coef(tp, tn, fp, fn, out_type=torch.float32, large_nums=True):
    # formula: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """
        if perfect_predictions(tp=tp, tn=tn, fp=fp, fn=fn):
            return 1
        elif all_incorrect_predictions(tp=tp, tn=tn, fp=fp, fn=fn):
            return -1
    """

    # define MCC on these failure cases
    perfect_predict_mask = perfect_predictions(tp=tp, tn=tn, fp=fp, fn=fn)
    all_incorrect_predict_mask = all_incorrect_predictions(tp=tp, tn=tn, fp=fp, fn=fn)

    # will be True everywhere that doesn't have perfect/worst posssible prediction
    # still possible to divide by 0
    user_define_mcc_behvr_mask = torch.logical_or(perfect_predict_mask, all_incorrect_predict_mask)
    where_to_divide = torch.logical_not(user_define_mcc_behvr_mask)

    mcc_numerator = (tp * tn) - (fp * fn)
    if not large_nums:
        # the argument to torch.sqrt can easily overflow for large graphs. Compute is elementwise.
        mcc_denom = torch.sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) )
    if large_nums or torch.any(torch.isnan(mcc_denom)):
        if not large_nums:
            print(f'WARNING: When computing MCC, got overflow. Attempting to recompute. Try reducing batch size.')
        # each column is a different sample, each row are the sum values (e.g. tp + fp)
        sums = torch.stack(((tp + fp) , (tp + fn) , (tn + fp) , (tn + fn)))

        a = torch.floor(torch.sqrt(sums))
        mcc_denom = torch.prod(a, dim=0) * torch.sqrt( torch.prod(sums/a**2, dim=0) )

        if not large_nums:
            if torch.any(torch.isnan(mcc_denom)):
                print(f'\tAttempted to recover, but still NAN in MCC computation')
            else:
                print(f'\tSuccesffuly recovered: no NAN values in MCC')

    out = torch.zeros_like(tp, dtype=out_type)
    out[perfect_predict_mask] = 1
    out[all_incorrect_predict_mask] = -1

    """
    if np.any(mcc_denom == 0):
        txt = "Undefined MCC: "
        if np.isclose(tp+fp, 0):
            txt += "0 Positive Preds,  "
        if np.isclose(tp + fn, 0):
            txt += "0 Actual Positives,  "
        if np.isclose(tn + fp, 0):
            txt += "0 Actual Negatives, "
        if np.isclose(tn + fn, 0):
            txt += "0 Negative Preds"
        raise ZeroDivisionError("MCC Denom is 0")
        """
    return torch.where(where_to_divide,  mcc_numerator/mcc_denom, out)
    #return torch.divide(mcc_numerator, mcc_denom, out=out, where=where_to_divide)


def precision(tp, fp, eps: float = 1e-12):
    return tp / (tp + fp + eps)


def recall(tp, fn, eps: float = 1e-12):
    return tp / (tp + fn + eps)


def f1(tp, fp, fn, eps: float = 1e-12):
    return tp / (tp + .5*(fp+fn) + eps)


def macro_f1(tp, tn, fp, fn, eps: float = 1e-12):
    return (f1(tp=tp, fp=fp, fn=fn, eps=eps) + f1(tp=tn, fp=fp, fn=fn, eps=eps))/2


def fdr(tp, fp, eps: float = 1e-12):
    return fp / (fp + tp + eps)


def fpr(tn, fp, eps: float = 1e-12):
    return fp / (fp + tn + eps)


def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr


"""
# From raw prediction and labels, compute tp/tn/fp/fn
def confusion_matrix_unsigned(y_hat, y, reduction_axes):
    if torch.is_tensor(y_hat) and torch.is_tensor(y):
        assert y_hat.shape == y.shape
        tp = torch.sum((y_hat == y) & (y != 0), dim=reduction_axes)
        tn = torch.sum((y_hat == y) & (y == 0), dim=reduction_axes)
        fp = torch.sum((y_hat != y) & (y == 0), dim=reduction_axes)
        fn = torch.sum((y_hat != y) & (y != 0), dim=reduction_axes)
    elif type(y_hat) == np.ndarray and type(y) == np.ndarray:
        assert len(y_hat) == len(y)
        tp = np.sum((y_hat == y) & (y != 0), axis=reduction_axes)
        tn = np.sum((y_hat == y) & (y == 0), axis=reduction_axes)
        fp = np.sum((y_hat != y) & (y == 0), axis=reduction_axes)
        fn = np.sum((y_hat != y) & (y != 0), axis=reduction_axes)
    else:
        raise ValueError(f'y_hat and y must be same type: y_hat {type(y_hat)}, y {type(y)}')

    return tp, tn, fp, fn
"""


def confusion_matrix(y_hat, y, reduction_axes):
    # ** anything non-zero will be considered an edge prediction/existance **
    # -> thus prediction of -1 for an edge 1 -> true positive prediction!
    assert y_hat.shape == y.shape
    #assert (y == 1).logical_or(y == -1).logical_or(y == 0).all(), 'only values y can take are {0, -1, 1}'
    y_zero, y_hat_zero = (y == 0), (y_hat == 0)
    y_nonzero, y_hat_nonzero = ~y_zero, ~y_hat_zero

    # must be same sign
    #tp_same_sign = torch.sum((y_hat == y) & y_nonzero, dim=reduction_axes)
    tp = torch.sum(y_hat_nonzero & y_nonzero, dim=reduction_axes)
    #assert torch.allclose(tp, tp_1)
    tn = torch.sum(y_hat_zero & y_zero, dim=reduction_axes)
    fp = torch.sum(y_hat_nonzero & y_zero, dim=reduction_axes)
    fn = torch.sum(y_hat_zero & y_nonzero, dim=reduction_axes)
    return tp, tn, fp, fn



def symmetric_classification_metrics(y_hat: torch.tensor, y: torch.tensor, threshold: float, auc_loop=False):
    # for symmetric (undirected) graphs without self-loops
    if y_hat.ndim == 3:
        y_hat, y = y_hat.squeeze(), y.squeeze()
    assert (y_hat.shape == y.shape), 'inputs must be same size'
    assert y_hat.ndim == 2, f'symmetric classification metrics only take vectorized adjacencies'
    num_graphs, num_possible_edges = y_hat.shape

    # TP = edge exists and predicted edge
    # FN = edge exists but predicted no edge
    # FP = edge dne    and predicted edge
    # FN = edge exist  and predicted no edge

    # Note that if the edge weights can take negative values, we follow the l2g convention that any non-zero ouput
    # (with magnitude greater than the threshold) represents an edge prediction.
    # all examples assume threshold == 0
    # e.g. if we predict -.34 for (i,j) but true value is 0.67 -> this is a true positive (TP) prediction
    # e.g. if we predict 0    for (i,j) and true value is 0    -> this is a true negative (TN) prediction
    # e.g. if we predict 0    for (i,j) but true value is !=0  -> this is a false negative (FN) prediction
    # e.g. if we predict !=0  for (i,j) but true value is 0    -> this is a false positive (FP) prediction

    # we maintain the sign of the prediction/edge but as of not it is NOT used
    # -> only-non-zero property is used in confusion matrix
    #print(f'\n\t y_hat.sign() {y_hat.sign().device}, y_hat.abs() {y_hat.abs().device} y.sign() {y.sign().device}, threshold {threshold.device}')
    #print(f'\t intermed product 1 {(y_hat.abs() > threshold).device}')
    #print(f'\t intermed product 2 {(y_hat.sign() * (y_hat.abs() > threshold)).device}')
    y_hat_s, y_s = y_hat.sign() * (y_hat.abs() > threshold), y.sign()# * (y.abs() > 0)
    y_hat_s, y_s = y_hat_s.to(torch.int), y_s.to(torch.int)

    tp, tn, fp, fn = confusion_matrix(y_hat=y_hat_s, y=y_s, reduction_axes=1)
    acc = accuracy(tp=tp, tn=tn, total=num_possible_edges)

    og_metrics = {
        'pr': precision(tp=tp, fp=fp),
        're': recall(tp=tp, fn=fn),
        'f1': f1(tp=tp, fp=fp, fn=fn),
        'macro_f1': macro_f1(tp=tp, tn=tn, fp=fp, fn=fn),
        'acc': acc, 'error': 1 - acc,
        'mcc': matthew_corr_coef(tp=tp, tn=tn, fp=fp, fn=fn)
    }

    ## l2g metrics ##
    edges_false, edges_no_pred = (y_s == 0), (y_hat_s == 0)
    edges_true, edges_pred = ~edges_false, ~edges_no_pred

    mismatches = torch.logical_xor(edges_true, edges_pred)
    fp_l2g = torch.sum(mismatches & edges_pred, dim=1)
    #assert torch.allclose(fp_l2g, fp)
    p = torch.sum(edges_pred, dim=1).to(torch.float32)
    t = torch.sum(edges_true, dim=1).to(torch.float32)
    f = (num_possible_edges - t).to(torch.float32) #len(edges_true) - T
    SHD = torch.sum(mismatches, dim=1).to(torch.float32)
    FDR = fp / p
    TPR = tp / t
    FPR = fp / f

    l2g_metrics = {
        # l2g - begin
        'FDR': FDR,
        'TPR': TPR,
        'FPR': FPR,
        'SHD': SHD,
        'T': t,
        'P': p
    }

    auc_metrics = {}
    if auc_loop:
        auc, aupr = torch.zeros_like(tp), torch.zeros_like(tp)
        for i in range(num_graphs):
            edge_true_i, edges_pred_i = edges_true[i, :], y_hat[i, :]
            # note we DONT threshold -> thresholding done inside ROC computation
            edges_pred_i = np.absolute(edges_pred_i.numpy())
            auc[i], aupr[i] = get_auc(edge_true_i.numpy(), edges_pred_i)
        auc_metrics = {'aupr': aupr, 'auc': auc}
    ###

    return {**og_metrics, **l2g_metrics, **auc_metrics}

"""
# Aggregate relevent metrics into dict
def classification_metrics(y_hat: torch.tensor, y: torch.tensor, threshold: float, non_neg: bool, graph_or_edge='graph'):
    # must add concept of non-negativity <<-->> only consider non-zero elements as edge
    #assert ('int' in str(y_hat.dtype) and 'int' in str(y.dtype)) or (y_hat.dtype == torch.bool and y.dtype == torch.bool), f'only take binary/integer inputs -> must threshold first'
    y_hat, y = y_hat.squeeze(), y.squeeze()
    assert (y_hat.shape == y.shape), 'inputs must be same size'

    if not non_neg:
        # first use threshold to remove small edge values (in pos or neg dir), then convert back to o.g. sign
        # signed metrics - other things (e.g. f1) not well defined when values have > 2 states
        y_hat_thresh = y_hat.clone()
        y_hat_thresh[y_hat_thresh.abs() < threshold] = 0.0
        reduction_axes = (1, 2) if y.ndim == 3 else 1
        signed_acc = (y.sign() == y_hat_thresh.sign()).to(y.dtype).mean(dim=reduction_axes)
        return {'acc': signed_acc, 'error': 1-signed_acc}

    # any y_hat values in [-thresh, thresh] -> 0
    y_hat, y = y_hat.sign() * (y_hat.abs() > threshold), y.sign()# * (y.abs() > 0)
    y_hat, y = y_hat.to(torch.int), y.to(torch.int)

    if y.ndim == 3:
        y, y_hat = adj2vec(y), adj2vec(y_hat)
    assert y_hat.ndim == y.ndim == 2
    batch_size, num_possible_edges = y_hat.shape
    num_graphs = batch_size

    if graph_or_edge == 'graph':
        reduction_axes = 1
        tp, tn, fp, fn = confusion_matrix(y_hat=y_hat, y=y, reduction_axes=reduction_axes)
        total = num_possible_edges

    elif graph_or_edge == 'edge':
        reduction_axes = 0
        tp, tn, fp, fn = confusion_matrix(y_hat=y_hat, y=y, reduction_axes=reduction_axes)
        total = num_graphs

    else:
        print(f'unrecognized graph_or_edge {graph_or_edge}\n')
        exit(2)

    # recall = true positive rate (tpr), precision = positive predictive value (ppv)
    # unsigned metrics
    acc = accuracy(tp=tp, tn=tn, total=total)
    return {'pr': precision(tp=tp, fp=fp), 're': recall(tp=tp, fn=fn), 'f1': f1(tp=tp, fp=fp, fn=fn),
            'macro_f1': macro_f1(tp=tp, tn=tn, fp=fp, fn=fn),
            'acc': acc, 'error': 1 - acc,
            'mcc': matthew_corr_coef(tp=tp, tn=tn, fp=fp, fn=fn)}
"""


def regression_metrics(y_hat: torch.tensor, y: torch.tensor, self_loops=False): # raw_adj=False):
    # self_loops: if self loops - take full difference between full adjacency matrices -> includes diagonal and double
    #  counts edges (in undirected graphs)
    # - if no self_loops -> should be using simplified adjacency -> vectorized form
    is_matrix_batch = y.shape[-1] == y.shape[-2]
    if y_hat.ndim == 3 and not is_matrix_batch:
        y_hat = y_hat.squeeze()
    if y.ndim == 3 and not is_matrix_batch:
        y = y.squeeze()
    assert (y_hat.shape == y.shape), 'inputs must be same size, subtraction operator broadcasts incorrectly if not'
    matrix_and_self_loops = (self_loops and is_matrix_batch) # if last two dims are equal -> matrix
    vectorized_and_no_self_loops = ((not self_loops) and not is_matrix_batch)
    assert matrix_and_self_loops or vectorized_and_no_self_loops, f'must include self-loops if using full-adjs for regression. {matrix_and_self_loops}, {vectorized_and_no_self_loops}'

    reduction_axes = (1, 2) if self_loops else 1
    diff = (y_hat-y)
    se, ae = torch.square(diff).sum(dim=reduction_axes), torch.abs(diff).sum(dim=reduction_axes)
    nse = se / (y ** 2).sum(dim=reduction_axes)
    #gmse = torch.sum((y_hat - y) ** 2, dim = -1) / torch.sum(y ** 2, dim = -1)
    #assert torch.allclose(nse, gmse), f'comparing l2g gmse computation...seems exactly the same'
    se_per_edge, ae_per_edge = torch.square(diff).mean(dim=reduction_axes), torch.abs(diff).mean(dim=reduction_axes)
    #nse_glad = 10*torch.log10(se.sum() / label_sizes.sum())
    """
    if not raw_adj:
        assert y_hat.ndim == y.ndim == 2
        reduction_axes = 1 if graph_or_edge=='graph' else 0
        mse, mae = torch.sum(se, dim=reduction_axes), torch.sum(ae, dim=reduction_axes)
        nmse = torch.divide(mse, torch.sum(y**2, dim=reduction_axes))
    else:
        assert y_hat.ndim == y.ndim == 3
        reduction_axes = (1, 2)
        mse, mae = torch.sum(se, dim=(1, 2)), torch.sum(ae, dim=reduction_axes)
        nmse = torch.divide(mse, torch.sum(y**2, dim=reduction_axes))
    """
    return {'se': se, 'ae': ae, 'nse': nse, '10_log_nse': 10*torch.log10(nse), 'se_per_edge': se_per_edge, 'ae_per_edge': ae_per_edge}
    #loss = torch.sum((y_hat - y) ** 2, dim = -1) / torch.sum(y ** 2, dim = -1)


def compute_metrics(y_hat: torch.tensor, y: torch.tensor, threshold, self_loops):
    # classification metrics DO NOT consider diagonal (i.e. no self-loops)
    if y_hat.ndim == 3:
        y_hat, y = y_hat.squeeze(), y.squeeze()
    assert y_hat.shape == y.shape
    if (not self_loops) and y.ndim == 3:
        y, y_hat = adj2vec(y), adj2vec(y_hat)

    return {**symmetric_classification_metrics(y_hat=y_hat, y=y, threshold=threshold),
            **regression_metrics(y_hat=y_hat, y=y, self_loops=self_loops)}

"""
def hinge_loss(y, y_hat, margin, per_edge=True, slope=1):
    # returns hinge_loss of each scan in tensor
    # FOR y in {0,+1} NOT {-1, +1}
    assert y.dtype == torch.bool
    if y_hat.ndim == 3:
        y_hat = y_hat.squeeze()
    if y.ndim == 2:
        y = y.squeeze()
    assert y.shape == y_hat.shape and y.ndim == 2, f'adjs must be in row form'
    loss_when_label_zero = torch.maximum(torch.zeros_like(y_hat), y_hat - margin) # assume all y_hat >= 0
    loss_when_label_one = torch.maximum(torch.zeros_like(y_hat), -y_hat + (1 - margin))
    hinge_loss = torch.where(condition=y, input=loss_when_label_one, other=loss_when_label_zero) # outputs input where true
    hinge_loss = slope*hinge_loss # more slope = more punishment of error

    return torch.mean(hinge_loss, dim=1) if per_edge else torch.sum(hinge_loss, dim=1)
"""
# returns hinge_loss of each scan in tensor
# FOR y in {0,+1} NOT {-1, +1}
def hinge_loss(y, y_hat, margin=0.2, already_bin=False, per_edge=True, slope=1):
    y_bin = y if already_bin else (y > 0)
    if torch.is_tensor(y_bin):
        assert y_bin.dtype == torch.bool
    else:
        assert y_bin.dtype == np.bool
    #assert (y.ndim == 3) and (y.shape[-2]==y.shape[-1]) #

    #assert (y_hat.ndim == 3) and (y_hat.shape[-2]==y_hat.shape[-1])
    loss_when_label_zero = torch.maximum(torch.zeros_like(y_hat), y_hat - margin) # assume all y_hat >= 0
    loss_when_label_one = torch.maximum(torch.zeros_like(y_hat), -y_hat + (1 - margin))
    hinge_loss = torch.where(condition=y_bin, input=loss_when_label_one, other=loss_when_label_zero) # outputs input where true
    hinge_loss = slope*hinge_loss # more slope = more punishment of errorprediction_metrics_for_each_subnetwork
    reduction_dims = (1, 2) if y.ndim == 3 else 1
    hinge_loss_per_scan = torch.sum(hinge_loss, dim=reduction_dims)

    # each hinge loss is now normalized by number of possible edges: hinge loss/# possible edges
    if per_edge:
        bs, N = y.shape[0:2]
        total_possible_edges = N * N - N  # ignore diagonals
        return hinge_loss_per_scan/total_possible_edges
    else:
        return hinge_loss_per_scan


def best_threshold_by_metric(y_hat: torch.tensor, y: torch.tensor, thresholds: torch.tensor, metric: str, device: str):
    # Given list of thresholds, see which one optimizes given metric
    assert metric in ['acc', 'error', 'f1', 'pr', 're', 'mcc'], f'given {metric}'
    if metric == 'error':
        metric = 'acc'
    values = torch.zeros(len(thresholds), device=device)
    for i, threshold in enumerate(thresholds):
        # v contains a metric value for each sample. Need to reduct
        values[i] = symmetric_classification_metrics(y_hat=y_hat, y=y, threshold=threshold)[metric].mean()
        #values[i] = torch.mean(classification_metrics(y_hat=(y_hat > threshold), y=(y > 0.0))[metric])

    # mcc will return nan if all predictions True or False. Handle this by not considering these cases.
    if metric == 'mcc':
        print(f'WARNING: should use nanargmax for mcc')
    return thresholds[torch.argmax(values)]


if __name__ == "__main__":

    torch.manual_seed(50)
    a = torch.randint(low=0, high=2, size=(2, 5))
    b = a.clone()
    b[0, 0] = b[1, 1] = 1
    tp, tn, fp, fn = confusion_matrix(a, b, reduction_axes=1)
    metrics = symmetric_classification_metrics(a.to(torch.bool), b.to(torch.bool), threshold=0.0)

    print(a,'\n', b)
    print(metrics)



    # metrics for uncertainty estimation
    num_nodes = 5
    corrupted_nodes = set(1,2)

    corrupted_nodes_mask = [(i in corrupted_nodes) for i in range(num_nodes)]
    corrupted_nodes_mask = torch.Tensor(corrupted_nodes_mask)

    zeros = torch.zeros(num_nodes, num_nodes)
    clean_edges, single_corrupt_edges, double_corrupt_edges, any_corrupt_edges \
        = zeros.__deepcopy__(), zeros.__deepcopy__(), zeros.__deepcopy__(), zeros.__deepcopy__()

    # costruct logical matrix which encodes the edge sets of interest
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i in corrupted_nodes and j in corrupted_nodes:
                # both have been corrupted
                double_corrupt_edges[i, j] = True
                any_corrupt_edges[i, j] = True
            elif i in corrupted_nodes or j in corrupted_nodes:
                # one one has been corrupted
                single_corrupt_edges[i, j] = True
                any_corrupt_edges[i, j] = True
            else:
                # neither corrupt
                clean_edges[i, j] = True

    clean_edges = clean_edges.logical_or(clean_edges.T)
    single_corrupt_edges = single_corrupt_edges.logical_or(single_corrupt_edges.T)
    double_corrupt_edges = double_corrupt_edges.logical_or(double_corrupt_edges.T)
    any_corrupt_edges = any_corrupt_edges.logical_or(any_corrupt_edges.T)

    # vectorize these matrices as in dpg/l2g
    clean_edges_vec = adj2vec(clean_edges.unsqueeze(0)).squeeze()


    # now compute MSE and mean stdv over each of these sets

    # reshape tensors for masking
    """
    corrupted_nodes_mask =  # mask tensor with indices in the set as True
    corrupted_nodes_mask = self.trainer.datamodule.perturbed_nodes.expand(batch_size, -1)


    # compute metrics using these masks
    mean_perturbed_se = (means - y).square()[perturbed_nodes_mask].sum(
        dim=1).mean()  # ensure shape after indexing correct
    mean_perturbed_corr = corrs.abs()[perturbed_nodes_mask].sum(dim=1).mean()

    mean_unperturbed_se = (means - y).square()[~perturbed_nodes_mask].sum(dim=1).mean()
    mean_perturbed_corr = corrs.abs()[perturbed_nodes_mask].sum(dim=1).mean()
    """


    # test cases
    # se over non corrupt = 1, se over corrupt = 4

