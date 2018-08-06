import numpy as np
import numpy.linalg as la
import timeit

import sys
from matplotlib import pyplot as plt
from matplotlib import ticker

import scipy as sc
import scipy.linalg as scl

import functools
import time

import ot
import torch
from gradients import gradient_chol, grad_AD_double


def normalize(v):
    return v / sum(v)


def create_my_plot(ad_time, gd_time, dim1, dim2):
    FF = lambda a, b:  a/b

    ad_time[ad_time == 0] = 9
    gd_time[gd_time == 0] = 9
    print("ad_time".format(np.mean(ad_time, axis=0)))
    print("".format(np.mean(gd_time, axis=0)))

    plt.imshow(sc.mean(FF(ad_time, gd_time), axis=0), cmap=plt.cm.get_cmap('RdBu'), vmin=-9, vmax=9)
    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_yticklabels(['1/8', '1/6', '1/4', '1/2', '1', '2', '4', '6', '8'])
    labely = [str((dim1[len(dim1)-1-i])) for i in range(len(dim1))]
    labelx = [str(dim1[i]) for i in range(len(dim2))]
    plt.xticks(np.arange(len(dim2)+1), labelx)
    plt.yticks(np.arange(len(dim1)+1), labely)
    plt.title('time ratio')

    plt.show()

    GG = lambda a, b: sc.log(a / b)

    plt.figure()
    plt.imshow(sc.mean(sc.log(ad_score), axis=0), cmap=plt.cm.get_cmap('Reds'), vmin=-18, vmax=0)
    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_yticklabels(['1e-8', '1e-6', '1e-4', '1e-2', '1'])
    labely = [str(dim1[i]) for i in range(len(dim1))]
    labelx = [str(dim1[i]) for i in range(len(dim2))]
    plt.xticks(np.arange(len(dim2)+1), labelx)
    plt.yticks(np.arange(len(dim1)+1), labely)
    plt.title('accuracy')
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run regime test to produce paper plots.')
    parser.add_argument('-n', type=int, nargs='+', default=(200, 1000, 2500, 5000, 10000, 25000, 30000),
                        help='Dim 1 list of numbers')
    parser.add_argument('-m', type=int, nargs='+', default=(200, 500, 1000),
                        help='Dim 2 list of numbers')
    parser.add_argument('-nr_seeds', type=int, default=5,
                        help='How many seeds')
    parser.add_argument('-iter_AD', type=int, default=30,
                        help='How many interation of Sinkhorn')
    parser.add_argument('-reg', type=float, default=0.02,
                        help='regularization')
    args = parser.parse_args()

    dim1 = args.n
    dim2 = args.m
    seeds = list(range(995, 995 + args.nr_seeds))

    print(seeds)
    # sys.exit()
    gd_time = np.zeros((len(seeds), len(dim1), len(dim2)))
    ad_time = np.zeros((len(seeds), len(dim1), len(dim2)))
    gd_score = np.zeros((len(seeds), len(dim1), len(dim2)))
    ad_score = np.zeros((len(seeds), len(dim1), len(dim2)))

    for idx_seed in range(len(seeds)):
        seed = seeds[idx_seed]
        np.random.seed(seed)

        reg = args.reg
        L_GD = 100
        L_AD = args.iter_AD
        tresh = 1e-11
        for idx_n, n in enumerate(dim1):
            for idx_m, m in enumerate(dim2):
                M = sc.spatial.distance.cdist(np.reshape(np.linspace(0, 1, n), (n, 1)),
                                              np.reshape(np.linspace(0, 1, m), (m, 1)),
                                              'sqeuclidean')

                a = normalize(np.random.rand(n))
                b = normalize(np.random.rand(m))

                if m > n:
                    continue
                print('n ', n)
                print('m ', m)
                true_grad = gradient_chol(a, b, M, reg, 1e5, 1e-11)

                # compute true gradient
                v = gradient_chol(a, b, M, reg, L_GD, tresh)

                # accuracy for GD
                gd_score[idx_seed, len(dim1) - idx_n - 1, idx_m] = la.norm(true_grad - v, 2)

                # compute time for GD with timeit
                t = timeit.Timer((functools.partial(gradient_chol, a, b, M, reg, L_GD, tresh)))
                time_detected = t.timeit(number=100) / 100
                print(time_detected)

                gd_time[idx_seed, len(dim1) - idx_n - 1, idx_m] = time_detected

                tM = torch.DoubleTensor(M)
                tb = torch.DoubleTensor(b)
                # tM = torch.from_numpy(M).type(dtype=torch.double)
                # tb = torch.from_numpy(b).type(dtype=torch.double)

                v = grad_AD_double(a, tb, tM, reg, L_AD, tresh)[2]

                # accuracy for AD
                v = v.data.numpy()
                ad_score[idx_seed, len(dim1) - idx_n - 1, idx_m] = la.norm(true_grad - v, 2)

                # compute time for AD with timeit
                t = timeit.Timer((functools.partial(grad_AD_double, a, tb, tM, reg, L_AD, tresh)))
                time_detected = t.timeit(number=100) / 100
                print(time_detected)

                ad_time[idx_seed, len(dim1) - idx_n - 1, idx_m] = time_detected

    print('done, going to create plots')
    create_my_plot(ad_time, gd_time, dim1, dim2)
    plt.show()
