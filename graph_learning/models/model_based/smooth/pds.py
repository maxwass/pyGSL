import pytorch_lightning
import torch, math, numpy as np, os, sys, time
from pathlib import Path
from torch.nn import functional as F
from typing import Dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file = Path(__file__).resolve()
path2project = str(file.parents[4]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> dpg/

import graph_learning.misc.metrics
from graph_learning.misc.utils import num_edges2num_nodes, matmulsq, set_accelerator
from graph_learning.models.model_based.iterative_base import iterative
from graph_learning.models.model_based.smooth.smooth import smooth_objective, primal_value

#import torch_sparse
#from torch_sparse import SparseTensor


"""
    PDS was introduced by  Kalofolias's "How to learn a graph from smooth signals", https://arxiv.org/abs/1601.02513
    
    It was unrolled by Pu in "Learning to Learn Graph Topologies". The implementation below borrow heavily from their 
    released version, found at their github repo: https://github.com/xpuoxford/L2G-neurips2021/blob/master/src/models.py

    Note on differences between PDS and unrolled PDS in Pu's work:
        - proximal log barrier steps seems to be different. In model based formulation, they do not clamp. 
            In unrolling, they do, and it seems to be important for their performance. This is not discussed in the paper.

"""


class PDS(iterative):
    """
        Proximal Dual Splitting algorithm to solve min_{A} ||A*E||_1 + \alpha * log(A1) + \beta*||A||_F^2
        Proposed by Vassilis Kalofolias in "How to learn a graph from smooth signals" https://arxiv.org/pdf/1601.02513.pdf
        Implemented by Max Wasserman, maxw14k@gmail.com

    """
    def __init__(self):
        super().__init__()


    @staticmethod
    def _setup(x, D, device):
        batch_size, num_edges = x.shape
        n = num_edges2num_nodes(num_edges)

        #initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, num_edges), device=device).float()
        v = torch.zeros((batch_size, n), device=device).float()

        return w, v

    @staticmethod
    def _primal_value(w: torch.tensor, x: torch.tensor, S: torch.tensor, alpha: float, beta: float):
        # note the authors of l2g scale the ||w e ||_1 term by 2. We exclude this.
        return primal_value(w=w, x=x, S=S, alpha=alpha, beta=beta)

    @staticmethod
    def _dual_value(w: torch.tensor, e: torch.tensor, S: torch.tensor, params):
        pass

    @staticmethod
    def _prox_log_barrier_dong(y, gamma, alpha, eps=1e-8):
        up = y ** 2 + 4 * gamma * alpha
        # CLAMPING (placing floor below values) CRITICAL FOR UNROLLING PERFORMANCE, but not included (or needed?) in model based
        up = torch.clamp(up, min=eps) if eps is not None else up
        return (y - torch.sqrt(up)) / 2


    @staticmethod
    def _shared_step_kal(z: torch.tensor,
                         w: torch.tensor,
                         d: torch.tensor,
                         S: torch.tensor,
                         alpha: float,
                         beta: float,
                         gamma: float,
                         params: Dict):
        """
        Algorithm 1 in "How to learn a graph from smooth signals"
        Implementation: Max Wasserman, maxw14k@gmail.com

        Does not converge, see below.

        NOTE: There are typo's in Algorithm 1 in "How to learn a graph from smooth signals":
            0) 2*z term excluded from y_i computation. Note its use at the end with: w = w - y1 + q1
            1) Line 6, sign of sqrt flipped (????)
            2) Line 7, (i) 2*z data term missing, (ii) used of p_i, correct: \bar{p}_i
            3) Line 9, use of p_i, correct: q_i
        """
        # alpha <-> log barrier on degrees, beta <-> l2 norm on edge weights, gamma <--> step size
        if not gamma: #'step_size' in params:
            gamma = compute_gamma(n=d.shape[-1], beta=beta, step_size=params['step_size'])
        S_t = S.transpose(-1, -2)
        # expand size of d from bs x N -> bs x N x 1 to make S.T d well defined:
        # S_t \in [bs, N(N-1)/2, N], d \in [bs, N] -> d must have shape [bs, N, 1] -> S_t*d \in [bs, N, 1]. Squeeze last dim.
        y1 = w - gamma * (2*beta*w + matmulsq(S_t, d))
        y2 = d + gamma * matmulsq(S, w)

        p1 = torch.relu(y1 - 2*gamma*z)
        p2 = 0.5*(y2 + torch.sqrt(y2 ** 2 + 4 * alpha * gamma)) # TYPO 1: outer + was -

        q1 = p1 - gamma*(2*beta*p1 + matmulsq(S_t, p2)) # TYPO 2: p2 was p1
        q2 = p2 + gamma*matmulsq(S, p1)

        w = w - y1 + q1 # TYPO 3: used of p_1, correct: q_1
        d = d - y2 + q2

        return w, d


    @staticmethod
    def _shared_step_dong(y: torch.tensor,
                          w: torch.tensor,
                          v: torch.tensor,
                          D: torch.tensor,
                          alpha: float,
                          beta: float,
                          gamma: float,
                          eps=None,
                          batch_first_dim: bool = True,
                          use_relu: bool = False):
        """
        Implemented by Max Wasserman, maxw14k@gmail.com. Insipred by Dong's L2G implementation.
            y is the input data -> called z in o.g. l2g code
            w is the refined graph estimate
            v is dual variable in R^N -> called d in Kalofolias "How to learn a graph from smooth signals"
            D is linear transform mapping vectorized adjacency into degree vector.
            alpha - the penalty before log barrier
            beta - the penalty before l2 term
            gamma - step size
        """
        # note that due to broadcasting under the hood of torch.matmul(), the following are equivalent.
        #   D.T v = torch.matmul(v, D)) = v.matmul(D)
        #   D w = torch.matmul(w, D.T) =  w.matmul(D.T)
        #   D.T p2 = torch.matmul(p2, D)) = p2.matmul(D)
        #   D p1 = torch.matmul(p1, D.T) = p1.matmul(D.T)

        #if type(D) == torch_sparse.tensor.SparseTensor:
        #    return PDS._shared_step_dong_sparse(y, w, v, D, params, eps)

        # alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
        batch_size, num_edges = w.shape
        _, m = v.shape
        z = y

        y1 = w - gamma * (2 * beta * w + 2 * z + v.matmul(D)) #torch.matmul(v, D)) # == D.T v
        y2 = v + gamma * w.matmul(D.T) # torch.matmul(w, D.T) # == D w

        p1 = torch.relu(y1) #
        p2 = PDS._prox_log_barrier_dong(y=y2, gamma=gamma, alpha=alpha, eps=eps) # prox_log_barrier DIFFERENT from model based

        q1 = p1 - gamma * (2 * beta * p1 + 2 * z + p2.matmul(D)) #torch.matmul(p2, D)) # == D.T*p2
        q2 = p2 + gamma * torch.matmul(p1, D.T) # == D*p1

        w = w - y1 + q1
        v = v - y2 + q2

        return w, v

    """
    @staticmethod
    def _shared_step_dong_sparse(y: torch.tensor,
                                 w: torch.tensor,
                                 v: torch.tensor,
                                 D: SparseTensor,
                                 params: Dict,
                                 eps=None):

        alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
        batch_size, num_edges = w.shape
        _, m = v.shape
        z = y

        # sparse operations
        spmm = torch_sparse.matmul
        sp_t = torch_sparse.t # sparse transpose

        #print(f'in pds model based: devices: w {w.device} z {z.device} v {v.device} D {D.device()}')
        y1 = w - gamma * (2 * beta * w + 2 * z + spmm(src=sp_t(D), other=v.unsqueeze(-1)).squeeze())
        y2 = v + gamma * spmm(src=D, other=w.unsqueeze(-1)).squeeze()

        p1 = torch.relu(y1)  #
        p2 = PDS._prox_log_barrier_dong(y=y2, gamma=gamma, alpha=alpha, eps=eps)

        q1 = p1 - gamma * (2 * beta * p1 + 2 * z + spmm(src=sp_t(D), other=p2.unsqueeze(-1)).squeeze())
        q2 = p2 + gamma * spmm(src=D, other=p1.unsqueeze(-1)).squeeze()

        w = w - y1 + q1
        v = v - y2 + q2

        return w, v
    """
    @staticmethod
    def _shared_step(x: torch.tensor,
                     w_prev: torch.tensor,
                     v_prev: torch.tensor,
                     D: torch.tensor,
                     n,
                     alpha: float,
                     beta: float,
                     gamma: float,
                     eps: float = 1e-16,
                     mimo: bool = False,
                     use_relu: bool = False
                     ):
        """
        Implemented by Max Wasserman, maxw14k@gmail.com. Insipred by Dong's L2G implementation.
            x is the input data -> called z in o.g. l2g code
            w_prev is the refined graph estimate
            v_prev is dual variable in R^N -> called d in Kalofolias "How to learn a graph from smooth signals"
            D is linear transform mapping vectorized adjacency into degree vector.
            alpha - the penalty before log barrier
            beta - the penalty before l2 term
            gamma - step size

        Different from dong shared_step: mimo-compatible, and uses pow(0.5) instead of torch.sqrt() due to bug found in latter
        """
        batch_size, num_edges = w_prev.shape

        y1 = w_prev - gamma * (2 * beta * w_prev + 2 * x + v_prev.matmul(D))  # torch.matmul(v, D)) # == D.T v
        y2 = v_prev + gamma * w_prev.matmul(D.T)  # torch.matmul(w, D.T) # == D w

        p1 = torch.relu(y1)  #

        # CLAMPING (placing floor below values) CRITICAL FOR UNROLLING PERFORMANCE, but not included (or needed?) in model based
        up = y2 ** 2 + 4 * gamma * alpha
        up = torch.clamp(up, min=eps) if eps is not None else up
        p2 = (y2 - up.pow(0.5)) / 2

        q1 = p1 - gamma * (2 * beta * p1 + 2 * x + p2.matmul(D))  # torch.matmul(p2, D)) # == D.T*p2
        q2 = p2 + gamma * torch.matmul(p1, D.T)  # == D*p1

        w = w_prev - y1 + q1
        v = v_prev - y2 + q2

        return w, v

    def solve(self,
              x: torch.tensor,
              labels: torch.tensor,
              alpha: float,
              beta: float,
              gamma: float,
              max_iter: int,
              eps_clamp: float,
              eps_convergence: float,
              logger=None,
              sparse: bool = False):
        assert x.ndim == 2
        batch_size, num_edges = x.shape
        m = num_edges2num_nodes(num_edges)
        assert alpha >= 0 and beta >= 0 and gamma > 0, f'hyperparameters must be non-negative'

        # move tensors to GPU if available
        device = torch.device(set_accelerator())
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.clone().to(device)
        labels = labels.clone().to(device)

        D = graph_learning.misc.utils.sumSquareForm(N=m, out_sparse=sparse).to(device) #l2g_utils.coo_to_sparseTensor(l2g_utils.get_degree_operator(m)).to(device)
        if sparse:
            raise Exception("torch sparse not installed!")
            #D = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(D)
        w, v = self._setup(x, D, device)

        i, stop, compute_time = 0, False, 0.0
        while i < max_iter and not stop:
            start_iter = time.time()
            w_prev, v_prev = w, v
            w_custom, v_custom = self._shared_step(x=x, w_prev=w, v_prev=v, n=m, D=D, alpha=alpha, beta=beta, gamma=gamma, eps=eps_clamp)
            #w, v = self._shared_step(x=x, w_prev=w, v_prev=v, n=m, D=D, alpha=alpha, beta=beta, gamma=gamma)
            w, v = self._shared_step_dong(y=x, w=w, v=v, D=D, alpha=alpha, beta=beta, gamma=gamma, eps=eps_clamp)
            print(f" diff w {(w-w_custom).abs().max():.3e}, v {(v-v_custom).abs().max():.3e}")

            #w, v = self._shared_step_dong(y=x, w=w, v=v, D=D_sparse, params=params, device=device)
            end_iter = time.time()
            iter_time = end_iter - start_iter
            compute_time += iter_time

            primal_vals_dict, primal_vals_raw = self._primal_value(w=w, x=x, S=D, alpha=alpha, beta=beta)
            metrics = self._compute_metrics(y_hat=w, y=labels)
            variable_change_dict = {
                'w': ((w - w_prev) ** 2).sum(dim=1).sqrt() / (w_prev ** 2).sum(dim=1).sqrt(),
                'v': ((v - v_prev) ** 2).sum(dim=1).sqrt() / (v_prev ** 2).sum(dim=1).sqrt()
            }
            stop, stop_dict = self._stopping_conditions(d=variable_change_dict, eps=eps_convergence)

            if logger:
                logger.log({**metrics, **stop_dict, **primal_vals_dict, 'iter_time': iter_time, 'total_time': compute_time}, step=i)
                for sample_idx, val in enumerate(primal_vals_raw):
                    logger.log({f'objective_val/{sample_idx}': val}, step=i)

            # stopping/divergence checks
            if metrics['nse/mean'] > 1e3:
                print(f"step {i}: nse has diverged - mean = {metrics['nse/mean']:.5f} >> 1.0...Terminating")
                break
            if stop:
                print(f'\tSTOP: all optimization variables have changed (l2-norm) less than {eps_convergence:.8f} in step {i}')
            if w.isnan().all() or v.isnan().all():
                stop = True
                print(f'\tSTOP: optimization variables have become NAN')
            print(f"\tstep {i} ({iter_time:.4f} s): nse/mean {metrics['nse/mean']:.5f}, primal_val/mean: {primal_vals_dict['primal_vals/mean']:.5f}, w.max()/w.min(): {w.max():.5f}/{w.min():.5f}, v.max()/v.min(): {v.max():.5f}/{v.min():.5f}")

            i += 1

        return w


if __name__ == '__main__':
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    from graph_learning.data.smooth_signals.common_sampling import graph_sampling
    from graph_learning.data.smooth_signals.dong_smooth_signals import smooth_signals
    config = {'graph_distribution': 'geom', 'weighted': True, 'N': 20, 'sigma': math.sqrt(0.1),
              'alpha': 2, 'beta': 2, 'gamma': .1,
              'max_iter': 10000, 'eps': 1e-6,
              'seed': 50}
    config = dotdict(config)
    graph_sampling_params = graph_sampling(graph_gen=config.graph_distribution, N=config.N, weighted=config.weighted)
    dm = smooth_signals(num_signals=3000, sigma=config.sigma,  # sigma=1e1,
                        graph_sampling_params=graph_sampling_params,
                        label='adjacency',  # 'precision', #'laplacian',
                        train_size=5, val_size=1, test_size=1,  # 4000, 1000, 100
                        batch_size=5,
                        num_workers=0,
                        seed=config.seed
                        )
    dm.setup()

    x, y = dm.train_dataloader().dataset.dataset.tensors
    pds = PDS()
    y_hat = pds.solve(x=x, labels=y,
                      alpha=config.alpha, beta=config.beta, gamma=config.gamma,
                      max_iter=config.max_iter,
                      eps_clamp=None, eps_convergence=config.eps,
                      logger=None)

