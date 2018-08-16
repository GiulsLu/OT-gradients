import numpy as np
import scipy.linalg as scl

import time
import copy
import ot
import torch


__all__ = ['grad_AD_double', 'gradient_chol']


def grad_AD_double(a, b, M, reg, niter, tresh):
    """Gradient with automatic differentiation."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64) / n
    v = torch.ones(m, dtype=torch.float64) / m

    K = torch.exp(-M / reg)

    x = torch.tensor(a, dtype=torch.double, requires_grad=True)

    Kp = (1/x).view(n, 1) * K
    cpt = 0
    err = 1
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K, 0, 1), u.view(n, 1))
        v = b / KtransposeU.view(m)
        u = 1. / torch.mm(Kp, v.view(m, 1))

        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    cost.backward()
    grad = x.grad
    grad_norm = grad - torch.dot(torch.ones(n, dtype=torch.float64), grad) * torch.ones(n, dtype=torch.float64) / n
    return T, cost, grad_norm


def gradient_chol(a, b, M, reg, numIter, tresh):
    """Compute gradient with closed formula using cholesky factorization."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    #T = ot.sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=numIter,
                #    stopThr=tresh)
    T = Tpytorch(a, b, M, reg, numIter, tresh)
    # if m < n we use Sherman Woodbury formula
    if m < n:
        D1i = 1/(np.sum(T, axis=1))
        D2 = np.sum(T[:, 0:m-1], axis=0)
        L = T*M
        f = -np.sum(L, axis=1) + T[:, 0:m-1] @ ((np.sum(L[:, 0:m-1].T, axis=1)) / D2)
        grada = D1i * f
        TDhalf = np.multiply(T[:, 0:m - 1].T, np.sqrt(D1i))
        K = np.diag(D2) - TDhalf @ TDhalf.T

        Lchol = scl.cho_factor(K+1e-15*np.eye(K.shape[0]), lower=True)

        grada = grada + D1i * (T[:, 0:m-1] @ scl.cho_solve(Lchol, T[:, 0:m-1].T @ grada))

    else:
        D1 = np.sum(T, axis=1)
        D2i = 1 / (np.sum(T[:,0:m-1], axis=0))
        L = T * M
        f = -np.sum(L, axis=1) + T[:, 0:m - 1] @ ((np.sum(L[:, 0:m - 1].T, axis=1)) * D2i)
        TDhalf = np.multiply(T[:, 0:m - 1], np.sqrt(D2i))
        K = np.diag(D1) - TDhalf @ TDhalf.T

        Lchol = scl.cho_factor(K + 1e-15 * np.eye(K.shape[0]), lower=True)

        grada = scl.cho_solve(Lchol, f)

    grada = -(grada - np.ones(n) * np.dot(grada, np.ones(n)) / n)
    return grada



def Tpytorch(a, b, M, reg, niter, tresh):
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64)/n
    v = torch.ones(m, dtype=torch.float64)/m

    tM=torch.DoubleTensor(M)
    K = torch.exp(-tM/reg)

    x = torch.tensor(a, dtype=torch.double, requires_grad=False)
    y = torch.DoubleTensor(b)

    Kp = (1/x).view(n,1) * K
    cpt = 0
    err = 1

    while (err > tresh and cpt < niter):
        uprev = u
        vprev = v
        KtransposeU = torch.mm(torch.transpose(K,0,1),u.view(n,1))
        v = y / KtransposeU.view(m)
        u = 1./torch.mm(Kp, v.view(m,1))

        if (KtransposeU == 0).any() or torch.isnan(u).any()\
             or torch.isnan(v).any() or (u == float('inf')).any():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.sum( (u - uprev) ** 2)/ torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2)/torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    T = T.data.numpy()
    TT = copy.deepcopy(T)
    return TT
