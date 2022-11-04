"""
    This file contains small adjustments to the code written by Subhransu Maji, mostly to make it work with more
    recent version of python and pytorch. See author attribution below.

    Matrix square root and its gradient on the GPU, https://people.cs.umass.edu/~smaji/projects/matrix-sqrt/
    Author: Subhransu Maji (smaji@cs.umass.edu)
    Date: Dec 19, 2017

    In the future it may pay to look into more recent methods which claim better performance,
    e.g. "FAST DIFFERENTIABLE MATRIX SQUARE ROOT", https://github.com/KingJamesSong/FastDifferentiableMatSqrt

"""
import argparse, torch, numpy as np, time as tm, scipy
from torch.autograd import Variable
from scipy.linalg import sqrtm as sqrtm_scipy

from torch import bmm, sqrt, mean, sum


# Compute error
def compute_error(A, sA):
    normA = sqrt(sum(sum(A * A, dim=1), dim=1))
    error = A - bmm(sA, sA)
    error = sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return mean(error)


# Forward + Backward via SVD decomposition
def sqrt_svd_lyap(A, dldz, dtype):
    # z=sA, dldz = d(loss)/d(sA)
    # we aim to compute sA and dlda. We are given dldz by autograd.
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    dlda = torch.zeros(batchSize, dim, dim, device=A.device).type(dtype)
    sA = torch.zeros(batchSize, dim, dim, device=A.device).type(dtype)
    for i in range(batchSize):
        U, S, V = (A[i, :, :].data).svd()
        sA[i, :, :] = (U.mm(S.diag().sqrt())).mm(V.t())
        S = S.diag().sqrt().mm(torch.ones(dim, dim).type(dtype))
        IU = U.t()
        dlda[i, :, :] = -U.mm(((IU.mm(dldz[i, :, :].data)).mm(IU.t()))/ (S + S.t())).mm(U.t())
    return sA, dlda, compute_error(A, Variable(sA, requires_grad=False))


# Forward via Denman-Beavers iterations
def sqrt_denman_beavers(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    sA = torch.zeros(batchSize, dim, dim, device=A.device).type(dtype)
    for n in range(batchSize):
        Y = (A[n, :, :]).data
        Z = torch.eye(dim, dim).type(dtype)
        for i in range(numIters):
            Y_ = 0.5 * (Y + Z.inverse())
            Z = 0.5 * (Z + Y.inverse())
            Y = Y_
        sA[n, :, :] = Y
    sA = Variable(sA, requires_grad=False)
    error = compute_error(A, sA)
    return sA, error


# Forward via Newton-Schulz iterations
# Backward via autograd
def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = Variable(torch.eye(dim, dim, device=A.device).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)
    Z = Variable(torch.eye(dim, dim, device=A.device).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    return sA, error


# Forward via Newton-Schulz iterations (non autograd version)
# Seems to be slighlty faster and has much lower memory overhead
def sqrt_newton_schulz(A, numIters, dtype):
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim, dim, device=A.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    Z = torch.eye(dim, dim, device=A.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    return sA, error


# Backward via iterative Lyapunov solver
def lyap_newton_schulz(z, dldz, numIters, dtype):
    batchSize = z.shape[0]
    dim = z.shape[1]
    normz = z.mul(z).sum(dim=1).sum(dim=1).sqrt()
    a = z.div(normz.view(batchSize, 1, 1).expand_as(z))
    I = torch.eye(dim, dim, device=a.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    q = dldz.div(normz.view(batchSize, 1, 1).expand_as(z))
    for i in range(numIters):
        q = 0.5 * (q.bmm(3.0 * I - a.bmm(a)) - a.transpose(1, 2).bmm(a.transpose(1, 2).bmm(q) - q.bmm(a)))
        a = 0.5 * a.bmm(3.0 * I - a.bmm(a))
    dlda = 0.5 * q
    return dlda


def scipy_sqrtm(A):
    sA = torch.zeros_like(A, device=A.device)
    for i in range(len(A)):
        sA[i] = torch.tensor(sqrtm_scipy(A[i].detach().numpy().astype(np.float_)), device=A.device)
    error = compute_error(A, sA)
    return sA, error


def scipy_grad_sylv(sA, dldz):
    grad_sA = torch.zeros_like(sA, device=sA.device)
    for i in range(len(sA)):
        sA_i = sA[i].detach().numpy().astype(np.float_)
        dldz_i = dldz[i].detach().numpy().astype(np.float_)
        grad_sA[i] = torch.tensor(scipy.linalg.solve_sylvester(sA_i, sA_i, dldz_i), device=sA.device)
    return grad_sA


# Create random PSD matrix
def create_symm_matrix(batchSize, dim, numPts, tau, dtype):
    A = torch.zeros(batchSize, dim, dim).type(dtype)
    for i in range(batchSize):
        pts = np.random.randn(numPts, dim).astype(np.float32)
        sA = np.dot(pts.T, pts) / numPts + tau * np.eye(dim).astype(np.float32)
        A[i, :, :] = torch.from_numpy(sA)
    print(f'Creating batch {batchSize}, dim {dim}, pts {numPts}, tau {tau:.3f}, dtype {dtype}') # % (batchSize, dim, numPts, tau, dtype))
    return A


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Matrix squareroot and its gradient demo')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pts', type=int, default=1000, metavar='N',
                        help='number of points to construct covariance matrix (default: 1000)')
    parser.add_argument('--tau', type=float, default=1.0, metavar='N',
                        help='conditioning by adding to the diagonal (default: 1.0)')
    parser.add_argument('--num-iters', type=int, default=10, metavar='N',
                        help='number of schulz iterations (default: 5)')
    parser.add_argument('--dim', type=int, default=100, metavar='N',
                        help='size of the covariance matrix (default: 64)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    if args.cuda:
        d = torch.cuda.FloatTensor
    else:
        d = torch.FloatTensor

    # Create matrix and gradient randomly
    np.random.seed(50)
    # A is input into sqrt function
    A = Variable(create_symm_matrix(batchSize=args.batch_size, dim=args.dim, numPts=args.pts, tau=args.tau, dtype=d),
                 requires_grad=True)
    # dldz is the gradient of loss wrt sqrt ~output~ <= given to us by autograd
    dldz = Variable(torch.randn(args.batch_size, args.dim, args.dim).type(d), requires_grad=False)
    dldz = 0.5 * (dldz + dldz.transpose(1, 2))

    # >>> Forward + backward with SVD >>>
    # Time: O(n^3), Space: O(n^3)
    print('Singular Value Decomposition (SVD):')
    start = tm.time()
    svd_sA, svd_grad, svd_error = sqrt_svd_lyap(A.detach(), -dldz, dtype=d)
    end = tm.time()
    svd_time = end - start
    print(f'  >> forward + backward time {svd_time:.5f}, forward error {svd_error.data:.2e}')
    # <<<

    # >>> Forward pass with Denman-Beavers iterations (no backward) >>>
    print(f'Denman-Beavers iterations ({args.num_iters} iters)')
    start = tm.time()
    db_sA, error = sqrt_denman_beavers(A.detach(), args.num_iters, dtype=d)
    end = tm.time()
    print(f'  >> forward time {end-start:.5f}, error {error.data:.2e}')
    print(f'  >> no backward via autograd')
    # <<<

    # >>> Forward pass with Newton-Schulz (autograd version) >>>
    # Time: O(Tn^2), Space: O(Tn^2), with T iterations
    print(f'Newton-Schulz iterations ({args.num_iters} iters)')
    start = tm.time()
    ns_auto_sA, error = sqrt_newton_schulz_autograd(A, args.num_iters, dtype=d)
    end = tm.time()
    iter_time = end - start
    print(f'  >> forward: time {end-start:.5f}, error {error.data:.2e}')

    # Backward pass with autograd
    start = tm.time()
    ns_auto_sA.backward(dldz)
    end = tm.time()
    iter_time += end - start
    backward_error = svd_grad.dist(A.grad.data)
    print(f'  >> backward via autograd: time {end - start:.5f}, error (vs SVD) {backward_error:.2e}') # % (end - start, backward_error))
    print(f'  >> speedup over SVD: {(svd_time / iter_time):.2f}') # % (svd_time / iter_time))
    # <<<

    # >>> Forward pass with Newton-Schulz >>>
    # Time: O(Tn^2), Space: O(n^2), with T iterations
    print(f'Newton-Schulz iterations (foward + backward) ({args.num_iters} iters) ')
    start = tm.time()
    ns_sA, error = sqrt_newton_schulz(A.data, args.num_iters, dtype=d)
    end = tm.time()
    iter_time = end - start
    print(f'  >> forward: time {end-start:.5f}, error {error:.2e}')
    # <<<

    # >>> Backward pass with Newton-Schulz
    start = tm.time()
    lyap_dlda = lyap_newton_schulz(ns_sA, dldz.data, args.num_iters, dtype=d)
    end = tm.time()
    iter_time += end - start
    backward_error = svd_grad.dist(lyap_dlda)
    print(f'  >> backward: time {end-start:.5f}, error (vs SVD) {backward_error:.2e} ')
    print(f'  >> speedup over SVD: {(svd_time / iter_time):.2f}')
    # <<<

    # >>> Forward pass with scipy sqrtm: Blocked Schur Algorithms for Computing the Matrix Square Root
    print(f'Scipy (foward)')
    start = tm.time()
    sp_sA, error = scipy_sqrtm(A.detach())
    end = tm.time()
    iter_time = end - start
    print(f'  >> forward: time {end - start:.5f}, error {error:.2e}')
    # <<<

    # >>> Backward pass with scipy solving sylverster equations >>>
    print('Scipy Sylvesters Equations (backward)')
    start = tm.time()
    sp_dlda = scipy_grad_sylv(svd_sA, dldz)
    end = tm.time()
    iter_time = end - start
    backward_error = svd_grad.dist(sp_dlda)
    print(f'  >> backward: time {end - start:.5f}, error (vs SVD) {backward_error:.2e}')
    # <<<
