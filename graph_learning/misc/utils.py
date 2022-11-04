"""
    Utility functions useful for graph structure learning methods
"""

from __future__ import print_function, division
import numpy as np, torch
from math import sqrt
from typing import List
#import torch_sparse

def set_accelerator():
    if torch.has_mps:
        return 'mps'
    elif torch.has_cuda:
        return 'cuda'
    else:
        return 'cpu'

def sample_spherical(npoints, ndim=3, rand_seed=78):
    np.random.seed(rand_seed)
    # each column is a point on the unit n-sphere
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def edge_density(A: torch.tensor):
    # any non-negative entry is considered an edge. No-self loops. Only undirected.
    if A.ndim == 2:
        bs, total_edges = A.shape
        assert bs != total_edges, f'Single matrix not allowed. Please leading add dimension'
        A_abs = A.abs() if A.dtype != torch.bool else A
        return (A_abs > 0).sum(dim=1)/total_edges
    elif A.ndim == 3:
        assert A.shape[-1] == A.shape[-2]
        A_vec = adj2vec(A)
        A_vec_abs = A_vec.abs() if A_vec.dtype != torch.bool else A_vec
        bs, total_edges = A_vec.shape
        return (A_vec_abs > 0).sum(dim=1) / total_edges


def sumSquareForm(N: int, dtype=np.int32, out_dtype=torch.float32, out_sparse=False):
    # construct linear transformation that computes the degree vector from vectorized adjacency matrix
    # (of unidrected graphs with no self loops)
    # size = N x N(N-1)/2
    NumCol = np.array(N * (N - 1) / 2, dtype=dtype)
    I = np.zeros(NumCol, dtype=dtype)
    J = np.zeros(NumCol, dtype=dtype)
    k = 0
    for i in np.arange(N - 1):
        I[k:k + N - i - 1] = np.arange(i + 1, N)
        k = k + N - i - 1
    k = 0
    for i in np.arange(N - 1):
        J[k:k + N - i - 1] = i
        k = k + N - i - 1
    Jidx = np.concatenate((I, J), axis=0)
    Iidx = np.tile(np.arange(NumCol, dtype=dtype), 2)
    if out_sparse:
        indices = np.stack((Iidx, Jidx))
        values = np.ones_like(Iidx, dtype=dtype)
        St = torch.sparse_coo_tensor(indices, values, dtype=out_dtype)
        return torch.transpose(St, dim0=0, dim1=1)
    else:
        St = np.zeros((NumCol, N), dtype=dtype) #if not out_sparse else torch.sparse_coo_tensor(size = (NumCol, N), dtype=dtype)
        St[Iidx, Jidx] = 1
        S = St.T
        return torch.tensor(S, dtype=out_dtype)

def matmulsq(a: torch.tensor, b: torch.tensor):
    # makes code cleaner. Likely fancy way to get into correct vectorized form without it.
    return a.matmul(b.unsqueeze(dim=-1)).squeeze()


def vec2adj(v: torch.tensor, n: int):
    # take batch of vectorized adjacencies
    # return stack of adjacency matrices

    if v.ndim==3:
        v = v.squeeze()
    assert v.ndim == 2
    batch_size, num_possibe_edges = v.shape
    adjs = torch.zeros(batch_size, n, n, device=v.device)

    # note: the vectorization occurs row-wise, and so we copy contiguos elements in v into row
    counter = 0 # index into v for each loop
    for row in range(n):
        numElements2Copy = n-row-1
        adjs[:, row, (row+1):] = v[:, counter:(counter+numElements2Copy)]
        counter += numElements2Copy

    return adjs + torch.transpose(adjs, dim0=-1, dim1=-2)


def adj2vec(A: torch.tensor, offset=1):
    # offset = 1 : by default dont include diagonal
    assert A.ndim == 3 and (A.shape[-1] == A.shape[-2]), f'adj2vec: input is not stack of matrices: input shape {A.shape}'
    batch_size, n = A.shape[:2]
    idxs = torch.triu_indices(n, n, offset=offset) # ignore diagonal
    v = A[:, idxs[0], idxs[1]].clone()
    return v


def num_edges2num_nodes(num_edges: int):
    assert num_edges>2
    # https://www.wolframalpha.com/input/?i=solve+y%3D+%28x*%28x-1%29%2F2%29
    inv = sqrt(8*num_edges+1)+1
    num_nodes = int(0.5*inv)
    assert num_nodes*(num_nodes-1)//2 == num_edges
    return num_nodes

    # Differentiable cov/corr operations
    # https://github.com/pytorch/pytorch/issues/19037


# cov and corr funcs
def batch_cov(points):
    # https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
    assert points.ndim == 3, f'batch_cov: input data must be of shape [batch_size, num_variables, num_observations]'
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def corrFromCov(bcov):
    assert bcov.ndim == 3 and bcov.shape[-1] == bcov.shape[-2], f'batch of covariance matrices'
    # formula https://stats.stackexchange.com/questions/413033/how-to-express-a-correlation-matrix-in-terms-of-a-covariance-matrix
    # D = diag(Cov)^-1/2
    # corr = diag(D) * cov * diag(D)
    D = bcov.diagonal(dim1=1, dim2=2).rsqrt()
    D_ = D.diag_embed(dim1=1, dim2=2)
    bcorr = D_.bmm(bcov).bmm(D_)  # torch.bmm(D_, bcov).bmm(D_)
    return bcorr


def batch_corr(points):
    return corrFromCov(batch_cov(points))


def cov2corr_torch(cov):
    og_shape = cov.shape
    if len(cov.shape) == 2:
        cov = cov.repeat(1, 1, 1)

    batch_size, N, _ = cov.shape

    # the correlation matrix = D*cov*D = diag(covs)^(-.5)*covs*diag(covs)^(-.5)
    # https://en.wikipedia.org/wiki/Covariance_matrix#Relation_to_the_correlation_matrix
    D = torch.zeros((batch_size, N, N), dtype=cov.dtype)
    diag_in_rows = torch.diagonal(cov, dim1=1, dim2=2)
    diag_in_rows_t = 1 / torch.sqrt(diag_in_rows)
    for row in range(batch_size):
        # diagonal matrix D's diagonal entires is the row of diag_in_rows_t
        D[row] = torch.diag(diag_in_rows_t[row])
    corr = torch.matmul(torch.matmul(D, cov), D)
    return corr.view(og_shape)  # matrix vs batch 1 tensor


def cov2corr_np(cov):
    """
    assert len(cov.shape) == 2, f'have not yet handled tensor of covs for numpy'
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    correlation = cov / outer_v
    correlation[cov == 0] = 0
    return correlation
    """
    og_shape = cov.shape
    # enforces 3D tensor
    if len(cov.shape) == 2:
        cov = np.expand_dims(cov, axis=0)
        cov = np.repeat(cov, 1, axis=0)

    assert cov.shape[1] == cov.shape[2], f'cov must be square matrices'
    batch_size, N, _ = cov.shape
    D = np.zeros((batch_size, N, N), dtype=cov.dtype)
    diag_in_rows = np.diagonal(cov, axis1=1, axis2=2)
    diag_in_rows_t = 1 / np.sqrt(diag_in_rows)
    for row in range(batch_size):
        # diagonal matrix D's diagonal entires is the row of diag_in_rows_t
        D[row] = np.diag(diag_in_rows_t[row])
    corr = np.matmul(np.matmul(D, cov), D)

    # ensure output has same shape as input (we expanded dim for matrix input)
    corr_view = corr.view()
    # https://stackoverflow.com/questions/11524664/how-can-i-tell-if-numpy-creates-a-view-or-a-copy/14271298#14271298
    corr_view.shape = og_shape  # will throw error if incompatible
    return corr_view


def correlation_from_covariance(covariance):
    if torch.is_tensor(covariance):
        return cov2corr_torch(covariance)
    else:
        return cov2corr_np(covariance)


def adjs2fvs(adjs_list: List[torch.tensor]):
    # given list of batched (possibly multichannel) adjs, create elementwise feature vectors
    # ex) given 3 batched adjs a, b, c \in (bs x n x n) output fv \in {3 x bs*n*n }
    # once done, an mlp outputs o \in { bs*n*n }. Call a .view(bs, n, n) to return to o.g. shape
    try:
        return torch.cat([a.view(-1, 1) for a in adjs_list], dim=1)
    except:
        return torch.cat([a.reshape(-1, 1) for a in adjs_list], dim=1)



mp = "matrix polynomial:"
t = "torch.Tensor"
def matrix_polynomial(S, coeffs):
    # convert back to numpy at end if input is numpy
    nparrays = isinstance(S, np.ndarray)
    np_dtype = S.dtype

    # convert both S and coeffs to tensors
    if isinstance(S, np.ndarray):
        S = torch.from_numpy(S)
    dtype = S.dtype

    if not torch.is_tensor(coeffs):
        coeffs = torch.tensor(coeffs, dtype=dtype)

    #assert len(S.shape) == 3, f'matrix_polynomial: input As must have batch, not single matrix'
    assert len(coeffs.shape) == 1, f'{mp} coeffs must be a 1D {t}, e.g. (L,), is {coeffs.shape}'
    assert coeffs.shape[0] > 0, f'{mp} must have at least 1 coefficient'

    if len(S.shape) == 2:
        S = torch.unsqueeze(S, 0)

    num_matrices, N, _ = S.shape
    num_coeffs = coeffs.shape[0]

    # efficient powers of matrix
    #powers = torch.zeros((num_coeffs, num_matrices, N, N), dtype=dtype)
    powers = []
    # identities along all sub-matrices in batch
    #powers[0] = torch.eye(N).reshape((1, N, N)).repeat(num_matrices, 1, 1)
    powers.append(torch.eye(N).reshape((1, N, N)).repeat(num_matrices, 1, 1).type(dtype))

    for i in range(1, num_coeffs):
        # tensor multiplication: ith matrix in A multiplying ith in powers[i-1]
        #powers[i] = torch.matmul(S, powers[i - 1])
        powers.append(torch.bmm(S, powers[i - 1]))
    # tensordot should replace this i think
    # if powers is made into tensor, may be able to use matmal -> dot to scale respective matrices, them sum
    outputs = torch.zeros((num_matrices, N, N))
    for i in range(num_coeffs):
        outputs += coeffs[i] * powers[i]

    if nparrays:
        outputs = outputs.numpy().astype(np_dtype)

    return outputs


def tensor_powers(S: torch.tensor, polynomial_order):
    # S is assumed to be of shape [C_in, bs, N, N]
    assert len(S.shape) == 4
    assert S.shape[-2] == S.shape[-1]

    [_, batch_size, N, _] = S.shape

    # Construct tensor of shape [num_coeffs, C_in, bs, N, N], where each entry on 0th dim is a power of input
    powers_S = torch.empty((polynomial_order+1, *S.shape), dtype=S.dtype, device=S.device)#.type_as(S) # empty()

    # first power is identity matrix
    powers_S[0] = torch.broadcast_to(torch.eye(N), S.shape)
    if polynomial_order >= 1:
        powers_S[1] = S
    if polynomial_order >= 2:
        powers_S[2] = torch.matmul(S, S)
    if polynomial_order >= 3:
        powers_S[3] = torch.matmul(powers_S[2], S)
    if polynomial_order >= 4:
        powers_S[4] = torch.matmul(powers_S[3], S)
    if polynomial_order >= 5:
        powers_S[5] = torch.matmul(powers_S[4], S)
    if polynomial_order >= 6:
        powers_S[6] = torch.matmul(powers_S[5], S)
    if polynomial_order >= 7:
        powers_S[7] = torch.matmul(powers_S[6], S)
    if polynomial_order >= 8:
        powers_S[8] = torch.matmul(powers_S[7], S)
    if polynomial_order >= 9:
        powers_S[9] = torch.matmul(powers_S[8], S)
    if polynomial_order >=10:
        raise ValueError('Have not implimented polynomials of order >=6')
    """
    # When looping, get error:
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation, [torch.FloatTensor [1000, 68, 68]], which is output 0 of ViewBackward, is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
    # In order to be efficient for higher order polynomials, repeatedly multiply by S and store intermediate results
    for power in range(2, polynomial_order+1):
        powers_S[power] = torch.matmul(powers_S[power - 1], S)
    """

    return powers_S


# given C_in 3D tensors of slices (bsxNxN), output C_in 3D tensors (bsxNxN) of polynomial slices
def tensor_polynomial(input_channels, filter_params):
    # input channels shape = [C_in, bs, N, N]
    # filter params shape  = [order+1, C_in] <- specify polynomial to be used for EACH input channel
    assert (len(input_channels.shape) == 4) and (len(filter_params.shape) == 2)
    assert input_channels.shape[-1] == input_channels.shape[-2], f'input tensors slices must be square: {input_channels.shape}'
    C_in, bs, N = input_channels.shape[:-1]
    order = filter_params.shape[1] - 1  # num coeffs=order+1

    result = mimo_tensor_polynomial(input_channels, filter_params.unsqueeze(dim=0))

    return result.view(C_in, bs, N, N) # remove singleton C_out dimension


# Assume input is monomial basis
# COB monomial to chebyshev: https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
# https://en.wikipedia.org/wiki/Chebyshev_polynomials
def tensor_polynomial_COB_monomial2cheb(input_powers):
    # input_powers.shape [order+1, C_in, bs, N, N]
    # input_powers[ord] has C_in 3D tensors. Each is S_in^ord, where S_in is different for each input channel
    assert (len(input_powers.shape) == 5) and (input_powers.shape[-2] == input_powers.shape[-1])
    order_, C_in, bs, N, _ = input_powers.shape
    polynomial_order = order_ - 1

    tensor_powers_COB = torch.empty_like(input_powers)

    tensor_powers_COB[0] = input_powers[0]
    if polynomial_order >= 1:
        tensor_powers_COB[1] = input_powers[1]
    if polynomial_order >= 2:
        tensor_powers_COB[2] = 2 * input_powers[2] - input_powers[0]
    if polynomial_order >= 3:
        tensor_powers_COB[3] = 4*input_powers[3] - 3*input_powers[1]
    if polynomial_order >= 4:
        tensor_powers_COB[4] = 8*input_powers[4] - 8*input_powers[2] + input_powers[0]
    if polynomial_order >= 5:
        tensor_powers_COB[5] = 16*input_powers[5] - 20*input_powers[3] + 5*input_powers[1]
    if polynomial_order >= 6:
        tensor_powers_COB[6] = 32*input_powers[6] - 48*input_powers[4] + 18*input_powers[2] - input_powers[0]
    if polynomial_order >= 7:
        tensor_powers_COB[7] = 64*input_powers[7] - 112*input_powers[5] + 56*input_powers[3] - 7*input_powers[1]
    if polynomial_order >= 8:
        tensor_powers_COB[8] = 128*input_powers[8] - 256*input_powers[6] + 160*input_powers[4] - 32*input_powers[2] + input_powers[0]
    if polynomial_order >= 9:
        tensor_powers_COB[0] = 256*input_powers[9] - 576*input_powers[7] + 432*input_powers[5] - 120*input_powers[3] + 9*input_powers[1]
    if polynomial_order >= 10:
        raise ValueError('Have not implimented polynomials of order >=6')

    return tensor_powers_COB


def mimo_tensor_polynomial(input_channels, filter_params, cob='cheb', normalize_basis=None):
    # input channels shape = [C_in, bs, N, N]
    # filter params shape  = [C_out, order+1, C_in]
    # normalize_basis in [None, 'max_abs', ...] normalizes each basis term in matrix polynomial. Not currently used.
    assert (len(input_channels.shape) == 4) and (len(filter_params.shape) == 3) and (input_channels.shape[0] == filter_params.shape[-1])
    C_in, bs, N = input_channels.shape[:-1]
    C_out, order = filter_params.shape[0], filter_params.shape[1]-1 # num coeffs=order+1
    # basis to use for polynomial. Standard==None.
    assert cob in ['cheb', 'standard', None]

    # shape [order+1, C_in, bs, N, N]
    powers_in_fms = tensor_powers(input_channels, polynomial_order=order)

    if cob == 'cheb':
        # Change of Basis to ChebyShev
        powers_in_fms = tensor_polynomial_COB_monomial2cheb(powers_in_fms)

    if normalize_basis is not None:
        powers_in_fms = normalize_slices(powers_in_fms, which_norm=normalize_basis, extra='custom')

    # expand out for each output channel to have its own version to play with
    # shape = [C_out, order+1, C_in, bs, N, N]
    powers_in_fms_repeated = torch.broadcast_to(powers_in_fms, (C_out, *powers_in_fms.shape))
    assert torch.allclose(torch.tensor(powers_in_fms_repeated.shape), torch.tensor([C_out, order+1, C_in, bs, N, N]))

    # add singleton dimensions for proper broadcasting in coming multiplication
    # shape = [C_out, order+1, C_in, 1, 1, 1)
    fp_add_dims = filter_params.view(*filter_params.shape, 1, 1, 1)
    assert torch.allclose(torch.tensor(fp_add_dims.shape), torch.tensor([C_out, order+1, C_in, 1, 1, 1]))

    # scale power by appropriate coefficient in polynomials
    # shape = [C_out, order+1, C_in, bs, N, N]
    scaled_powers_in_fms_repeated = powers_in_fms_repeated * fp_add_dims
    assert torch.allclose(torch.tensor(scaled_powers_in_fms_repeated.shape), torch.tensor([C_out, order+1, C_in, bs, N, N]))

    # reduce out powers dimension => add up all terms in polynomials for a single tensor
    # shape = [C_out, C_in, bs, N, N]
    polynomial_in_fms_repeated = torch.sum(scaled_powers_in_fms_repeated, dim=1)
    assert torch.allclose(torch.tensor(polynomial_in_fms_repeated.shape), torch.tensor([C_out, C_in, bs, N, N]))

    return polynomial_in_fms_repeated


def symmetric_adj_normalize(a: torch.tensor, detach_norm: bool, undirected: bool = False):
    # FOR UNIDIRECTED GRAPHS DONT NEED INV SQRT!! SIMPLY DIVIDE BY POSITIVE DEGREES (take abs if negative)
    # only for undirected graphs
    # need to test for graphs with no edges ( 0 degree )
    #if type(a) == torch_sparse.tensor.SparseTensor:
    #    raise ValueError("given dense tensor, must reimplement for sparse. See PyG implementation")
    # D^{-1/2} A D^{-1/2}
    slice_shape = a.shape[-2:]
    assert len(slice_shape) == 2 and slice_shape[0] == slice_shape[1], f'symmetric adj normalization can only be applied to square matrices. Is {slice_shape} shaped.'

    # stack of slices
    batch_size, N = a.shape[0:2]
    slices = a.view(-1, *slice_shape)
    # should gradients flow through the degrees?
    slices = slices.detach() if detach_norm else slices

    ## BUG: diagonal entries should not be included here!! Some grapha have self loops (like covariance!)
    #print('symmetric norm: FIX BUG HERE')
    # dont include diagonal in degree
    #degrees = slices.sum(dim=-1) - torch.diag(slices)
    if undirected:
        # magintude of degrees
        degrees = slices.sum(dim=-1).abs()
        a_norm = a/degrees.view(batch_size, N, 1)

    else:

        degrees = slices.sum(dim=-1)
        #if (degrees<0).any():
        #    print(f'WARNING...NEGATIVE DEGREES!!')
        degrees[degrees < 0] = 0 # slices.sum(dim=-1).abs() # HACK...FIX THIS
        deg_inv_sqrt = degrees.pow(-0.5)

        with torch.no_grad():
            # if a degree was 0
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_mat = torch.diag_embed(deg_inv_sqrt) # ensure correct shape

        a_norm = deg_inv_sqrt_mat.bmm(a).bmm(deg_inv_sqrt_mat)
        # if undirected simply divide by degrees.abs()

    return a_norm.view(a.shape)


# we assume last two dimensions are the 'slices' to be normalized
def frob_normalize(x: torch.Tensor):
    slice_shape = x.shape[-2:]
    # THIS VIEW FAILS WHEN  C_out/C_in dims are SIMPLY broadcast into existance from a single batch. Need own memory?
    x_slices = x.view(-1, *slice_shape).detach()
    norms = torch.linalg.norm(x_slices, ord='fro', dim=(1, 2)).view(-1, 1)

    # first put into stack of slices, then into stack of rows
    x_row_form = x.view(-1, *slice_shape).view(x_slices.shape[0], -1)
    # first transpose to get shapes correct, second to bring back to og form
    x_row_form_normed = torch.div(x_row_form, norms)

    return x_row_form_normed.view(x.shape)


# we assume last two dimensions are the 'slices' to be normalized.
# We divide each slice by its OWN maximum abs value.
def max_abs_normalize(x: torch.Tensor, eps=1e-12):
    slice_shape = x.shape[-2:]
    x_slices = x.view(-1, *slice_shape).detach()
    norms = torch.amax(torch.abs(x_slices), dim=(1, 2)).view(-1, 1)

    # first put into stack of slices, then into stack of rows
    x_row_form = x.view(-1, *slice_shape).view(x_slices.shape[0], -1)
    # first transpose to get shapes correct, second to bring back to og form
    x_row_form_normed = torch.div(x_row_form, norms + eps)
    return x_row_form_normed.view(x.shape)


# we assume last two dimensions are the 'slices' to be normalized
def percentile_normalize(x: torch.Tensor, percentile, abs_vals=False, eps=1e-12):
    slice_shape = x.shape[-2:]
    x_slices = x.view(-1, *slice_shape).detach()

    x_in = x_slices.abs() if abs_vals else x_slices
    upper_tri_vals = adj2vec(x_in, offset=0)
    """
    if abs_vals:
        upper_tri_vals = upper_tri_as_vec_batch(torch.abs(x_slices), offset=0)
    else:
        upper_tri_vals = upper_tri_as_vec_batch(x_slices, offset=0)
    """
    # percentiles
    norms = torch.quantile(upper_tri_vals, percentile / 100, dim=-1, keepdim=True).view(-1, 1)

    # first put into stack of slices, then into stack of rows
    x_row_form = x.view(-1, *slice_shape).view(x_slices.shape[0], -1)
    # first transpose to get shapes correct, second to bring back to og form
    x_row_form_normed = torch.div(x_row_form, norms+eps)

    return x_row_form_normed.view(x.shape)


# https://stackoverflow.com/questions/62356695/pytorch-calculating-top-eigenvector-of-symmetric-matrix
# A.shape = [b, N, N]
# test similar matrices to find how many iterations will suffice
def TopEigenPair(A, n_power_iterations=15, eps=1e-8):
    # v.shape = [b, N, 1]
    v = torch.ones((A.shape[0], A.shape[1], 1), device=A.device)#.to(self.device)
    for _ in range(n_power_iterations):
        # m.shape = [b, N, 1]
        m = torch.bmm(A, v)
        # n.shape = [b]
        # Getting many nan values out of this. I believe this is because using 16 bit precision.
        # Smallest 16 bit number is ~1e-6. Was using eps ~ 1e-12.
        n = torch.sqrt(torch.sum(m**2, dim=1).unsqueeze(1) + eps)
        v = m / n

    # v is normalized. Eigenvalue = Av*v': https://ergodic.ugr.es/cphys/LECCIONES/FORTRAN/power_method.pdf
    evals = (torch.bmm(A, v) * v).sum(dim=1)

    if torch.any(torch.isnan(evals)):
      num_nan_evals = torch.isnan(evals).sum()
      num_total = A.shape[0]
      print(f'found eigenvalues are nan: {num_nan_evals}/{num_total}')
    return evals, v


#from scipy.sparse.linalg import eigs # doesnt work with batches?
# we assume all slice matrices are symmetric
def max_eig_normalize(x: torch.Tensor, which='symeig', eps=1e-12, niter=15):
    slice_shape = x.shape[-2:]
    x_slices = x.view(-1, *slice_shape).detach()

    with torch.no_grad():
        if which == 'symeig':
            # evals.shape = [*, N]. evals in ASCENDING order
            #evals_, evecs = torch.symeig(x_slices, eigenvectors=False) # deprecated
            evals = torch.linalg.eigvalsh(x_slices)
            #assert torch.allclose(torch.max(torch.abs(evals), dim=1)[0], torch.max(torch.abs(evals), dim=1)[0]) and evals.shape == evals_.shape
            norms = torch.max(torch.abs(evals), dim=1)[0].view(-1, 1)
        elif which == 'lobpcg':
            evals, evecs = torch.lobpcg(A=x_slices, k=1, largest=True, niter=niter)#, tol=10**-2)
            norms = evals.view(-1, 1)
        elif which == 'custom':
            # niter of 15 seems to converge quickly and accurately.
            evals, evecs = TopEigenPair(A=x_slices, n_power_iterations=niter)
            norms = evals.view(-1, 1)
        else:
            raise ValueError(f'Unrecognized eig method {which}')

    # first put into stack of slices, then into stack of rows
    x_row_form = x.view(-1, *slice_shape).view(x_slices.shape[0], -1)
    # first transpose to get shapes correct, second to bring back to og form
    x_row_form_normed = torch.div(x_row_form, norms + eps)
    return x_row_form_normed.view(x.shape)


# function which calls one of the above funcs
def normalize_slices(x: torch.tensor, which_norm='frob', extra='symeig', **kwargs):
    assert which_norm in ['symmetric_adj', 'frob', 'max_abs', 'percentile', 'max_eig', 'none', None, 'None']

    if which_norm in [None, 'none', 'None']:
        return x
    elif which_norm == 'symmetric_adj':
        return symmetric_adj_normalize(x, detach_norm=kwargs['detach_norm'])
    elif which_norm == 'frob':
        return frob_normalize(x) #frob_normalize_slices_batch(x)
    elif which_norm == 'max_abs':
        return max_abs_normalize(x) #max_abs_normalize_slices_batch(x)
    elif which_norm == 'percentile':
        raise ValueError('Fix extra value for this: feed in ~99')
        return percentile_normalize(x, percentile=extra, abs_vals=True) #percentile_normalize_slices_batch()
    elif which_norm == 'max_eig':
        if type(extra) == int or extra==None:
            raise ValueError(f'normalize slices using max_eig norm. Need to feed in proper extra. Given {extra}')
        elif type(extra) == 'str':
            #assert extra == 'symeig'
            print('using something other than symeig...test this!')
        return max_eig_normalize(x, which=extra)
    else:
        raise ValueError(f'which_norm unrecognized {which_norm}')


######################## TESTS ####################################
def test_matrix_polynomials():
    # [matrix_polynomial]
    # Module Responsibility: using 2D/3D tensor and polynomial coefficients to create a matrix polynomial
    # Assumptions:
    #  Inputs:
    #       # S is an (N,N)/(M,N,N numpy array
    #           # M, N >=1
    #       # coeffs is an (L,) numpy array
    #           # L>=1
    #  -correct order of matrix multiplications
    #  -temp matrix is correct

    mp = "matrix polynomial"
    # make sure matrix_polynomial asserts work
    # wrong size
    S = torch.zeros((2, 2))
    # H = matrix_polynomial(S,torch.ones(3))
    S = torch.zeros((2, 1, 1, 1))
    # H = matrix_polynomial(S,torch.ones(3))
    S = torch.zeros((2, 1, 1))
    # H = matrix_polynomial(S,torch.tensor([[1,1,1]]))

    # 1D
    c_111 = torch.ones(3)
    S = torch.zeros((2, 1, 1))
    S[0, :, :] = torch.eye(1)
    S[1, :, :] = torch.zeros(1)
    S_poly = torch.tensor([[[3]], [[1]]])
    H = matrix_polynomial(S, c_111)
    assert np.allclose(H, S_poly), f'{mp}: {H}'
    c_101 = torch.tensor([1, 0, 1])
    S_poly = torch.tensor([[[2]], [[1]]])
    H = matrix_polynomial(S, c_101)
    assert np.allclose(H, S_poly), f'{mp}: {H}'
    c_000 = torch.tensor([0, 0, 0])
    S_poly = torch.tensor([[[0]], [[0]]])
    H = matrix_polynomial(S, c_000)
    assert np.allclose(H, S_poly), f'{mp}: {H}'

    # 2D: identity
    # print('testing if matrix polynomial works for tensor of matrices')
    D = 2
    c_111 = torch.ones(3)
    S = torch.zeros((3, D, D))
    S[0, :, :] = torch.eye(D)
    S[1, :, :] = torch.zeros((D, D))
    cust = torch.tensor([[1, 1], [1, 0]])
    S[2, :, :] = cust
    # print(f'tensor to test:\n{S}')
    # print(f'coeffs to test:\n{c_111}')
    # input('inspect')
    S_poly = torch.zeros((3, D, D))
    S_poly[0, :, :] = 3 * torch.eye(D)
    S_poly[1, :, :] = torch.eye(D)
    S_poly[2, :, :] = torch.tensor([[4, 2], [2, 2]])
    # print(f'correct answer:\n{S_poly}')
    # input('inspect')
    H = matrix_polynomial(S, c_111)
    # print(f'output:\n{H}')
    # input('inspect')
    assert np.allclose(H, S_poly), f'{mp}: {H}'

    c_102 = torch.tensor([1, 0, 2])
    S = torch.zeros((3, D, D))
    S[0, :, :] = 2 * torch.eye(D)
    S[1, :, :] = torch.zeros((D, D))
    S[2, :, :] = cust
    S_poly = torch.zeros((3, D, D))
    S_poly[0, :, :] = 9 * torch.eye(D)
    S_poly[1, :, :] = torch.eye(D)
    S_poly[2, :, :] = torch.tensor([[5, 2], [2, 3]])
    H = matrix_polynomial(S, c_102)
    assert np.allclose(H, S_poly), f'{mp}: {H}'

    # 2D interesting matrix
    S = torch.tensor([[[1, 2], [-1, 2]]]).float()
    S_poly = torch.tensor([[[1, 8], [-4, 5]]]).float()
    H = matrix_polynomial(S, torch.ones(3))
    assert np.allclose(H, S_poly), f'{mp}: {H}'

    S_poly = torch.tensor([[[0, 6], [-3, 3]]])
    H = matrix_polynomial(S, torch.tensor([1, 0, 1]))
    assert np.allclose(H, S_poly), f'{mp}: {H}'

    #### different tests

    # 1D
    H = matrix_polynomial(np.eye(1), np.ones(3))
    H_float = matrix_polynomial(np.eye(1, dtype=np.float32), np.ones(3))
    assert np.allclose(H, np.array([[3.0]])), f'{mp}: {H}'
    assert np.allclose(H_float, np.array([[3.0]])), f'{mp}: {H}'
    H = matrix_polynomial(np.eye(1), np.array([1, 0, 1]))
    H_float = matrix_polynomial(np.eye(1, dtype=np.float32), np.array([1, 0, 1]))
    assert np.allclose(H, np.array([[2.0]])), f'{mp}: {H}'
    assert np.allclose(H_float, np.array([[2.0]])), f'{mp}: {H}'
    H = matrix_polynomial(np.eye(1), np.array([0, 0, 0.0]))
    H_float = matrix_polynomial(np.eye(1, dtype=np.float32), np.array([0, 0, 0.0]))
    assert np.allclose(H, np.array([[0.0]])), f'{mp}: {H}'
    assert np.allclose(H_float, np.array([[0.0]])), f'{mp}: {H}'

    # 2D: identity
    H = matrix_polynomial(np.eye(2), np.ones(3))
    H_float = matrix_polynomial(np.eye(2, dtype=np.float32), np.ones(3))
    assert np.allclose(H, 3 * np.eye(2)), f'{mp}: {H}'
    assert np.allclose(H_float, 3 * np.eye(2)), f'{mp}: {H}'
    H = matrix_polynomial(np.eye(2), np.array([1, 0, 1]))
    H_float = matrix_polynomial(np.eye(2, dtype=np.float32), np.array([1, 0, 1]))
    assert np.allclose(H, 2 * np.eye(2)), f'{mp}: {H}'
    assert np.allclose(H_float, 2 * np.eye(2)), f'{mp}: {H}'
    H = matrix_polynomial(np.eye(2), np.array([0, 0, 0.0]))
    H_float = matrix_polynomial(np.eye(2), np.array([0, 0, 0.0]))
    assert np.allclose(H, 0 * np.eye(2)), f'{mp}: {H}'
    assert np.allclose(H_float, 0 * np.eye(2)), f'{mp}: {H}'

    # 2D interesting matrix
    m = np.array([[1, 2], [-1, 2]])
    m_2 = np.matmul(m, m)
    H = matrix_polynomial(m, np.ones(3))
    H_float = matrix_polynomial(np.float32(m), np.ones(3))
    assert np.allclose(H, np.eye(2) + m + m_2), f'{mp}: {H}'
    assert np.allclose(H_float, np.eye(2) + m + m_2), f'{mp}: {H}'
    H = matrix_polynomial(m, np.array([1, 0, 1]))
    H_float = matrix_polynomial(np.float32(m), np.array([1, 0, 1]))
    assert np.allclose(H, np.eye(2) + m_2), f'{mp}: {H}'
    H = matrix_polynomial(m, np.array([0, 0.0, 0]))
    H_float = matrix_polynomial(np.float32(m), np.array([0, 0.0, 0]))
    assert np.allclose(H, 0 * np.eye(2)), f'{mp}: {H}'
    assert np.allclose(H_float, 0 * np.eye(2)), f'{mp}: {H}'

    ###### tests from summary_stats

    As = torch.zeros((2, 3, 3))
    As[0] = torch.tensor([[0, 1, 0], [1, 0, .5], [0, .5, 0]])
    As[1] = torch.tensor([[0, 0, .5], [0, 0, 1], [.5, 1, 0]])
    coeffs = torch.tensor([1, .5, 1])

    filters_soln = torch.zeros(2, 3, 3)
    for i in range(coeffs.shape[0]):
        filters_soln[0] = filters_soln[0] + torch.matrix_power(As[0], i) * coeffs[i]
        filters_soln[1] = filters_soln[1] + torch.matrix_power(As[1], i) * coeffs[i]

    filters = matrix_polynomial(As, coeffs)

    assert np.allclose(filters, filters_soln)


def test_mimo_tensor_polynomial():

    # MIMO matrix polynomials
    bs, order, N = 3, 2, 2
    dt = torch.float32
    slice_0 = torch.tensor([[1, 2], [0, 1]], dtype=dt)
    slice_1 = torch.tensor([[2, -1], [-1, 0]], dtype=dt)

    soln_slice0_coeffs123 = torch.tensor([[6, 16], [0, 6]], dtype=dt)
    soln_slice1_coeffs123 = torch.tensor([[20, -8], [-8, 4]], dtype=dt)
    soln_slice0_coeffs357 = torch.tensor([[15, 38], [0, 15]], dtype=dt)
    soln_slice1_coeffs357 = torch.tensor([[48, -19], [-19, 10]], dtype=dt)
    soln_slice0_coeffs123_bc = torch.broadcast_to(soln_slice0_coeffs123, (bs, *soln_slice0_coeffs123.shape))
    soln_slice1_coeffs123_bc = torch.broadcast_to(soln_slice1_coeffs123, (bs, *soln_slice1_coeffs123.shape))
    soln_slice0_coeffs357_bc = torch.broadcast_to(soln_slice0_coeffs357, (bs, *soln_slice0_coeffs357.shape))
    soln_slice1_coeffs357_bc = torch.broadcast_to(soln_slice1_coeffs357, (bs, *soln_slice1_coeffs357.shape))


    # IN 1, OUT 1
    C_out, C_in = 1, 1
    filter_params = torch.zeros((C_out, order+1, C_in))
    filter_params[0] = torch.tensor([[1.0], [2.0], [3.0]], dtype=dt)

    # slice 0
    in_fms = torch.broadcast_to(slice_0, (C_in, bs, *slice_0.shape))
    soln = soln_slice0_coeffs123_bc.view(1, 1, *soln_slice0_coeffs123_bc.shape)
    result = mimo_tensor_polynomial(in_fms, filter_params, cob='standard') #, channel_reduction_func=torch.mean)
    assert torch.allclose(soln, result) and torch.allclose(torch.tensor(soln.shape), torch.tensor(result.shape))
    # slice 1
    in_fms = torch.broadcast_to(slice_1, (C_in, bs, *slice_1.shape))
    soln = soln_slice1_coeffs123_bc.view(1, 1, *soln_slice1_coeffs123_bc.shape)
    result = mimo_tensor_polynomial(in_fms, filter_params, cob='standard') #, channel_reduction_func=torch.mean)
    assert torch.allclose(soln, result) and torch.allclose(torch.tensor(soln.shape), torch.tensor(result.shape))

    # IN 2, OUT 1
    C_in, C_out = 2, 1
    filter_params = torch.zeros((C_out, order + 1, C_in))
    filter_params[0, :, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=dt)
    filter_params[0, :, 1] = filter_params[0, :, 0]*2
    in_fms = torch.zeros((C_in, bs, N, N))
    in_fms[0] = torch.broadcast_to(slice_0, (bs, *slice_0.shape))
    in_fms[1] = torch.broadcast_to(slice_1, (bs, *slice_1.shape))
    soln = torch.zeros(C_out, C_in, bs, N, N)
    soln[0, 0] = soln_slice0_coeffs123_bc
    soln[0, 1] = 2*soln_slice1_coeffs123_bc
    result = mimo_tensor_polynomial(in_fms, filter_params, cob='standard') #, channel_reduction_func=torch.mean)
    assert torch.allclose(soln, result) and torch.allclose(torch.tensor(soln.shape), torch.tensor(result.shape))

    # IN 1, OUT 2
    C_in, C_out = 1, 2
    filter_params = torch.zeros((C_out, order + 1, C_in))
    filter_params[0] = torch.tensor([[1.0], [2.0], [3.0]], dtype=dt)
    filter_params[1] = filter_params[0]*2
    in_fms = torch.zeros((C_in, bs, N, N))
    in_fms[0] = torch.broadcast_to(slice_0, (bs, *slice_0.shape))
    soln = torch.zeros(C_out, C_in, bs, N, N)
    soln[0, 0] = soln_slice0_coeffs123_bc
    soln[1, 0] = 2*soln_slice0_coeffs123_bc
    result = mimo_tensor_polynomial(in_fms, filter_params, cob='standard') #, channel_reduction_func=torch.mean)
    assert torch.allclose(soln, result) and torch.allclose(torch.tensor(soln.shape), torch.tensor(result.shape))


    # IN 2, OUT 2
    C_in, C_out = 2, 2
    filter_params = torch.zeros((C_out, order + 1, C_in))
    filter_params[0, :, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=dt)
    filter_params[0, :, 1] = 2*filter_params[0, :, 0]
    filter_params[1, :, 0] = torch.tensor([3, 5, 7], dtype=dt)
    filter_params[1, :, 1] = 2*filter_params[1, :, 0]
    in_fms = torch.zeros((C_in, bs, N, N))
    in_fms[0] = torch.broadcast_to(slice_0, (bs, *slice_0.shape))
    in_fms[1] = torch.broadcast_to(slice_1, (bs, *slice_1.shape))
    soln = torch.zeros(C_out, C_in, bs, N, N)
    soln[0, 0] = soln_slice0_coeffs123_bc
    soln[0, 1] = 2*soln_slice1_coeffs123_bc
    soln[1, 0] = soln_slice0_coeffs357_bc
    soln[1, 1] = 2*soln_slice1_coeffs357_bc
    result = mimo_tensor_polynomial(in_fms, filter_params, cob='standard') #, channel_reduction_func=torch.mean)
    assert torch.allclose(soln, result) and torch.allclose(torch.tensor(soln.shape), torch.tensor(result.shape))


# helper func for test_batch_normalization
def normalize_test(slice, slice_frob_normed, slice_max_abs_normed, slice_percentile_normed, size_tensor, percentile):
    test_input = torch.broadcast_to(slice, (*size_tensor, *slice.shape))

    test_output_frob = frob_normalize(test_input)
    test_truth_frob = torch.broadcast_to(slice_frob_normed, (*size_tensor, *slice.shape))

    test_output_max_abs = max_abs_normalize(test_input)
    test_truth_max_abs = torch.broadcast_to(slice_max_abs_normed, (*size_tensor, *slice.shape))

    test_output_percentile = percentile_normalize(test_input, percentile, abs_vals=True)
    test_truth_percentile = torch.broadcast_to(slice_percentile_normed, (*size_tensor, *slice.shape))

    assert torch.allclose(test_output_frob, test_truth_frob)
    assert torch.allclose(test_output_max_abs, test_truth_max_abs)
    assert torch.allclose(test_output_percentile, test_truth_percentile)


def test_batch_normalization():

    slice = torch.tensor([[2.0, 2.0], [0, 1]])
    slice2 = torch.tensor([[-1, 1], [-2, -np.sqrt(3)]])
    slices = torch.zeros(2, 2, 2)
    slices[0], slices[1] = slice, slice2

    #### TEST 1 - single commone slice ####
    slice_frob_normed = slice/3
    slice_max_abs_normed = slice/2
    slice_percentile_normed = slice/1.5 # for 25th percentile <- remember only look at upper triangular part
    percentile = 25
    C_out, C_in, bs = 2, 3, 2
    size_tensor = (2, 3, 2)

    normalize_test(slice=slice, slice_frob_normed=slice_frob_normed,
                   slice_max_abs_normed=slice_max_abs_normed,
                   slice_percentile_normed=slice_percentile_normed,
                   size_tensor=size_tensor, percentile=percentile)


    #### TEST 2 - diff single common slice####
    slice2_frob_normed = slice2/3
    slice2_max_abs_normed = slice2/2
    slice2_percentile_normed = slice2/np.percentile([1, 1, np.sqrt(3)], percentile)

    normalize_test(slice2, slice2_frob_normed, slice2_max_abs_normed, slice2_percentile_normed, size_tensor, percentile)


    #### TEST 3 - stack of 2 slices ####
    # now combine slices together
    test_input = torch.zeros((C_out, C_in, bs, 2, 2), dtype=torch.float64)
    test_input[0][0], test_input[0][1], test_input[0][2] = torch.clone(slices), torch.clone(slices), torch.clone(slices)
    test_input[1][0], test_input[1][1], test_input[1][2] = torch.clone(slices), torch.clone(slices), torch.clone(slices)
    #test_input = torch.broadcast_to(slices, (C_out, C_in, *slices.shape))
    slices_frob_normed = torch.cat((slice_frob_normed.unsqueeze(dim=0), slice2_frob_normed.unsqueeze(dim=0)), dim=0)
    slices_max_abs_normed = torch.cat((slice_max_abs_normed.unsqueeze(dim=0), slice2_max_abs_normed.unsqueeze(dim=0)), dim=0)
    slices_percentile_normed = torch.cat((slice_percentile_normed.unsqueeze(dim=0), slice2_percentile_normed.unsqueeze(dim=0)), dim=0)

    combo_frob_soln = torch.broadcast_to(slices_frob_normed, (C_out, C_in, *slices.shape))
    combo_max_abs_soln = torch.broadcast_to(slices_max_abs_normed, (C_out, C_in, *slices.shape))
    combo_percentile_soln = torch.broadcast_to(slices_percentile_normed, (C_out, C_in, *slices.shape))

    assert torch.allclose(combo_frob_soln, frob_normalize(test_input))
    assert torch.allclose(combo_max_abs_soln, max_abs_normalize(test_input))
    assert torch.allclose(combo_percentile_soln, percentile_normalize(test_input, percentile, abs_vals=True))


    # eigenvalue normalization test
    slice_eig_1 = torch.tensor([[2.0, 2.0, 2], [2.0, 1.0, 3], [2, 3, 4]]).view(1, 3, 3)
    slice_eig_2 = torch.tensor([[-3, 1.0, 5], [2.0, 1.0, 3], [2, 3, 4]]).view(1, 3, 3)
    slice_eig_3 = torch.tensor([[-3, 1.0, 5], [2.0, 1.0, 8], [2, 8, 4]]).view(1, 3, 3)
    slices = torch.cat((slice_eig_1, slice_eig_2, slice_eig_3), dim=0)
    C_out, C_in, bs = 2, 3, 3
    test_input = torch.zeros((C_out, C_in, bs, 3, 3), dtype=torch.float64)
    test_input[0][0], test_input[0][1], test_input[0][2] = torch.clone(slices), torch.clone(slices), torch.clone(slices)
    test_input[1][0], test_input[1][1], test_input[1][2] = torch.clone(slices), torch.clone(slices), torch.clone(slices)
    slice_eig_1_normed = slice_eig_1/7.29841
    slice_eig_2_normed = slice_eig_2/7.13398
    slice_eig_3_normed = slice_eig_3/11.508
    slices_eig_normed = torch.cat((slice_eig_1_normed, slice_eig_2_normed, slice_eig_3_normed), dim=0)
    combo_eig_soln = torch.broadcast_to(slices_eig_normed, (C_out, C_in, *slices.shape))
    assert torch.allclose(combo_eig_soln.to(torch.float64), max_eig_normalize(test_input, which='lobpcg'))


    # symmetric adj normalization
    # 0<-->1<-->2
    a = torch.tensor([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]]).float()
    a1 = a.clone()
    a_self_loop = a + torch.eye(3)
    d_inv_sqrt = torch.diag(torch.tensor([2, 3, 2]).pow(-0.5))
    soln = d_inv_sqrt.mm(a_self_loop).mm(d_inv_sqrt)
    out = symmetric_adj_normalize(a_self_loop.unsqueeze(0), detach_norm=True)
    assert torch.allclose(out, soln)

    a = torch.tensor([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]]).float()
    a2 = a.clone()
    a_self_loop = a + torch.eye(3)
    d_inv_sqrt = torch.diag(torch.tensor([3, 3, 3]).pow(-0.5))
    soln = d_inv_sqrt.mm(a_self_loop).mm(d_inv_sqrt)
    out = symmetric_adj_normalize(a_self_loop.unsqueeze(0), detach_norm=True)
    assert torch.allclose(out, soln)

    a_self_loop = torch.stack((a1, a2)) + torch.diag(torch.ones(3)).expand(2, 3, 3)
    d_inv_sqrt = torch.diag_embed(torch.tensor([[2, 3, 2], [3, 3, 3]]).pow(-0.5))
    soln = d_inv_sqrt.bmm(a_self_loop).bmm(d_inv_sqrt)
    out = symmetric_adj_normalize(a_self_loop, detach_norm=True)
    assert torch.allclose(out, soln)



    #### RANDOM TESTS ###
    x = torch.randn(3, 10, 3, 3)
    x_frob_norm = frob_normalize(x).view(-1, 3, 3)
    x_inf_norm = max_abs_normalize(x).view(-1, 3, 3)

    for i in range(x_frob_norm.shape[0]):
        slice = x.view(-1, 3, 3)[i, :, :]
        assert np.allclose(slice / torch.linalg.norm(slice, ord='fro'), x_frob_norm[i, :, :]), 'frob norm fail'
        assert np.allclose(slice / torch.max(torch.abs(slice)), x_inf_norm[i, :, :]), 'inf norm fail'
    """ 
    ##### compare  NEW vs OLD implimentations (which are only for 3D) ####
    a = torch.rand(50, 3, 3)
    assert torch.allclose(frob_normalize(a), frob_normalize_slices_batch(a))
    assert torch.allclose(max_abs_normalize(a), max_abs_normalize_slices_batch(a))
    assert torch.allclose(percentile_normalize_slices_batch(a, 25, abs_vals=True), percentile_normalize(a, 25, abs_vals=True))

    ####### TESTS FOR OLD IMPLIMENTATIONS #######
    # percentile normalize batch tests
    a = torch.tensor(
        [[[1, 0.0], [0, 7.0]],
        [[3.0, -1.0], [-1.0, 8.0]]])

    a_norm_50 = torch.zeros_like(a)
    a_norm_50[0] = a[0]
    a_norm_50[1] = a[1]/3

    a_norm_75 = torch.zeros_like(a)
    a_norm_75[0] = a[0]/4
    a_norm_75[1] = a[1]/5.5

    assert torch.allclose(percentile_normalize_slices_batch(a, 50), a_norm_50)
    assert torch.allclose(percentile_normalize_slices_batch(a, 75), a_norm_75)


    # normalized adj test
    x = torch.zeros((2, 3, 3), dtype=torch.float32)
    x[0] = torch.tensor([
        [0, .5, 1],
        [.5, 0, 0],
        [1,  0, 0]
    ])
    x[1] = torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    soln = torch.zeros_like(x)
    soln[0] = torch.tensor([
        [0.0000, 0.5774, 0.8165],
        [0.5774, 0.0000, 0.0000],
        [0.8165, 0.0000, 0.0000]])
    soln[1] = torch.tensor([
        [0.0000, 0.5000, 0.5000],
        [0.5000, 0.0000, 0.5000],
        [0.5000, 0.5000, 0.0000]])

    #out = normalized_adj_batch(x)
    #assert torch.max(torch.abs(out-soln).view(-1))<.0001
    """


def test_custom_top_eigenpair():
    from data.network_diffusion import DiffusionDataset
    from data.brain_data import RealDataModule
    import time
    scs = DiffusionDataset(num_vertices=500, num_samples=2, fc_norm=None).full_ds()[1]
    #scs = RealDataModule(fc_norm=None).full_ds()[1]

    x = scs
    slice_shape = x.shape[-2:]
    x_slices = x.view(-1, *slice_shape).detach()

    evals_truth, evecs_truth = torch.symeig(x_slices, eigenvectors=False)
    largest_evals_truth = torch.max(torch.abs(evals_truth), dim=1)[0].view(-1, 1)

    for n_pow_iters in range(5, 25, 5):
        #largest_evals_approx, _ = TopEigenPair(A=x_slices, n_power_iterations=n_pow_iters)
        largest_evals_approx, evecs = torch.lobpcg(A=x_slices, k=1, largest=True, niter=n_pow_iters)  # , tol=10**-2)
        largest_error = torch.max(torch.abs(largest_evals_truth - largest_evals_approx))
        print(f'n_power_iterations: {n_pow_iters}, max error: {largest_error}')


def adjs2fvs_tests():
    torch.manual_seed(50)
    bs, n = 3, 2
    a = torch.rand(bs, n, n)
    # test 1: comparison happening entry-wise, so a & -a should produce 0
    fv = adjs2fvs([a, -a])
    out = fv.mean(dim=1).view(bs, n, n)
    soln = torch.zeros_like(a)
    torch.allclose(out, soln)

    # test 2: now confirm that resultant reshaping to (bs, n, n) is consistant with original.
    # -> idea is to transform (mean) in such a way to get a's values, and reshape to see if same
    #     layout as o.g. a
    fv = adjs2fvs([a, -a, 3*a])
    out = fv.mean(dim=1).view(bs, n, n)
    soln = a
    torch.allclose(out, soln)

    # test 3: now given multi-channel batch of adjs
    channels = 2
    a = torch.rand(channels, bs, n, n)
    out = adjs2fvs([a, -a]).mean(dim=1).view(channels, bs, n, n)
    torch.allclose(out, torch.zeros_like(a))

    out = adjs2fvs([a, -a, 3*a]).mean(dim=1).view(channels, bs, n, n)
    torch.allclose(out, a)


def num_edges2num_nodes_tests():
    for n in range(3, 50):
        num_edges = int(n*(n-1)/2)
        assert n == num_edges2num_nodes(num_edges)


def vec2adj_tests():
    # 3d
    torch.manual_seed(50)
    num_graphs, n = 10, 6
    # create symmetric stack of matrices
    a = torch.zeros(num_graphs, n, n)
    for g in range(num_graphs):
        b = torch.randint(0, 2, size=(n, n))
        b = b + torch.transpose(b, dim0=-1, dim1=-2)
        for i in range(len(b)):
            b[i, i] = 0
        a[g] = b

    # convert matrices to vectors
    v = torch.zeros(num_graphs, int(n*(n-1)/2))
    for i in range(num_graphs):
        idxs = torch.triu_indices(n, n, offset=1)
        v[i, :] = a[i, idxs[0], idxs[1]]

    # convert back to matrices
    a_hat = vec2adj(v, n)
    assert torch.allclose(a.to(a_hat.dtype), a_hat)


def adj2vec_tests():
    # 3d
    torch.manual_seed(50)
    num_graphs, n = 10, 6
    # create symmetric stack of matrices
    a = torch.zeros(num_graphs, n, n)
    for g in range(num_graphs):
        b = torch.randint(0, 2, size=(n, n))
        b = b + torch.transpose(b, dim0=-1, dim1=-2)
        for i in range(len(b)):
            b[i, i] = 0
        a[g] = b

    # convert matrices to vectors manually
    v_soln = torch.zeros(num_graphs, int(n*(n-1)/2))
    for i in range(num_graphs):
        idxs = torch.triu_indices(n, n, offset=1)
        v_soln[i, :] = a[i, idxs[0], idxs[1]].clone()

    # use our method
    v_out = adj2vec(a.clone())

    # should be the same
    assert torch.allclose(v_soln.to(v_out.dtype), v_out)

    # convert back to matrices, nothing should have change
    a_hat = vec2adj(v_out.clone(), n)
    assert torch.allclose(a.to(a_hat.dtype), a_hat)


def sumSquareForm_tests():
    ### 3 nodes ###
    # no edges
    a = torch.zeros((3,3)) # adj
    d = torch.zeros(3) # degree vec
    w = torch.zeros(3) # upper tri of adj
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # one edge
    a[0,1]=a[1,0]=1
    d[0] = d[1] = 1
    w[0] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # two edges
    a[0, 2] = a[2, 0] = 1
    d = torch.tensor([2, 1, 1.0])
    w[1] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # fully connected
    a[1,2] = a[2,1] = 1
    d = 2.0*torch.ones(3)
    w[2] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    ### 4 nodes ###
    n = 4
    w_len = int(n*(n-1)/2)
    # no edges
    a = torch.zeros((n, n))  # adj
    d = torch.zeros(n)  # degree vec
    w = torch.zeros(w_len)  # upper tri of adj
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # one edge
    a[0, 1] = a[1, 0] = 1
    d[0] = d[1] = 1
    w[0] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # two edges
    a[0, 2] = a[2, 0] = 1
    d = torch.tensor([2, 1, 1.0, 0])
    w[1] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # 3 edges
    a[1, 2] = a[2, 1] = 1
    d = torch.tensor([2.0, 2, 2, 0])
    w[3] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # 4 edges
    a[2, 3] = a[3, 2] = 1
    d = torch.tensor([2.0, 2, 3, 1])
    w[-1] = 1
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # fully connected: 6 edges
    a[0, 3] = a[3, 0] = 1
    a[1, 3] = a[3, 1] = 1
    d = 3.0 * torch.ones(n)
    w = torch.ones(w_len)
    assert torch.allclose(sumSquareForm(len(a)) @ w, d)

    # sparse tensors
    for n in [10, 100, 200, 250]:
        sparse_S = sumSquareForm(n, out_sparse=True)
        S = sumSquareForm(n)
        assert torch.allclose(S, sparse_S.to_dense().float())


def test_cov_corr():
    # COV/CORR TESTS
    # testing construction of, and transformations between, covariance
    #  matrices and correlation matrices in numpy and pytorch
    import time
    B = 10000
    N = 50
    D = 2
    points = torch.randn(B, N, D)

    start_cov = time.time()
    my_covs = batch_cov(points)
    print("Time for batched cov", time.time() - start_cov)

    start_corr = time.time()
    my_cors = batch_corr(points)
    print("Time for batched corr", time.time() - start_corr)

    start_cov = time.time()
    torch_covs = torch.zeros_like(my_covs)
    for i, batch in enumerate(points):
        torch_covs[i] = batch.T.cov()
    print("Time for looped Torch cov", time.time() - start_cov)

    start_corr = time.time()
    torch_corrs = torch.zeros_like(my_covs)
    for i, batch in enumerate(points):
        torch_corrs[i] = batch.T.corrcoef()
    print("Time for looped Torch corrcoef", time.time() - start_corr)

    print("Same?", torch.allclose(my_covs, torch_covs, atol=1e-7) and torch.allclose(my_cors, torch_corrs, atol=1e-7))





    x = torch.rand(3, 100)
    b = True
    # np.cov/corrcoef -> 2D np matrix
    # cov/corr -> 2D torch tensor
    # are direct computations of cov/corr same as numpy?
    correction = 0 if b else 1
    cov = torch.cov(x, correction=correction)
    assert np.allclose(np.cov(x, bias=b), cov) and np.allclose(np.corrcoef(x), torch.corrcoef(x))
    # is mapping from custom cov to corr same as true corr?
    assert np.allclose(correlation_from_covariance(cov), np.corrcoef(x))
    # same but now with numpy implimentation
    assert np.allclose(correlation_from_covariance(cov.numpy()), np.corrcoef(x))
    # is *mapping from custom torch cov to corr* same as *mapping from custom numpy cov to corr*
    assert np.allclose(correlation_from_covariance(cov), correlation_from_covariance(np.cov(x)))

    # now try tensors
    cov_tensor_gt = np.repeat(np.expand_dims(np.cov(x, bias=b), axis=0), 5, axis=0)
    cov_tensor = cov.repeat(5, 1, 1)
    # nothing new, just asserting that expaning batch dim worked
    assert np.allclose(cov_tensor, cov_tensor_gt)
    # is mapping from cov torch tensor -> corr torch tensor same as
    #  mapping from cov numpy tensor -> corr numpy tensor?
    assert np.allclose(correlation_from_covariance(cov_tensor), correlation_from_covariance(cov_tensor_gt))


if __name__ == "__main__":
    test_cov_corr()

    test_batch_normalization()

    adjs2fvs_tests()
    vec2adj_tests()
    adj2vec_tests()
    sumSquareForm_tests()
    num_edges2num_nodes_tests()
    test_cov_corr()


    test_matrix_polynomials()
    test_mimo_tensor_polynomial()
    #test_custom_top_eigenpair()
