import numpy as np
import numpy.linalg as la
import timeit


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


def create_my_plot(ad_time, gd_time, ad_score):
    FF = lambda a, b: sc.sign(a - b) * a/b

    ad_time[ad_time==0] = 9
    gd_time[gd_time==0] = 9
    print("ad_time".format(np.mean(ad_time, axis=0)))
    print("".format(np.mean(gd_time, axis=0)))

    plt.imshow(sc.mean(FF(ad_time, gd_time), axis=0), cmap=plt.cm.get_cmap('RdBu'), vmin=-9, vmax=9)
    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_yticklabels(['1/8', '1/6', '1/4', '1/2', '1', '2', '4', '6', '8'])
    plt.xticks(np.arange(6), ('200', '500', '1000', '2000', '5000', '10000'))
    plt.yticks(np.arange(7), ('30000', '20000', '10000', '5000', '2500', '1000', '200'))

    plt.show()

    GG = lambda a, b: sc.log(a / b)

    plt.figure()
    plt.imshow(sc.mean(sc.log(ad_score), axis=0), cmap=plt.cm.get_cmap('Reds'), vmin=-18, vmax=0)
    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_yticklabels(['1e-8', '1e-6', '1e-4', '1e-2', '1'])
    plt.xticks(np.arange(6), ('200', '500', '1000', '2000', '5000', '10000'))
    plt.yticks(np.arange(7), ('30000', '20000', '10000', '5000', '2500', '1000', '200'))


if __name__ == '__main__':

    #
    # dim1 = [200, 1000, 2500, 5000, 10000, 25000]
    # dim2 = [200, 500, 1000, 2000, 5000, 10000]

    dim1 = [200, 1000, 2500, 5000, 10000, 20000, 30000]
    dim2 = [200, 500, 1000, 2000, 5000, 10000]
    seeds = [995, 996, 997, 998, 999]
    gd_time = np.zeros((len(seeds), len(dim1), len(dim2)))
    ad_time = np.zeros((len(seeds), len(dim1), len(dim2)))
    gd_score = np.zeros((len(seeds), len(dim1), len(dim2)))
    ad_score = np.zeros((len(seeds), len(dim1), len(dim2)))

    for idx_seed in range(len(seeds)):
        seed = seeds[idx_seed]
        np.random.seed(seed)

        reg = 0.02
        L_GD = 100
        L_AD = 30
        
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

                #compute true gradient
                v = gradient_chol(a, b, M, reg, L_GD, tresh)

                # accuracy for GD
                gd_score[idx_seed, len(dim1) - idx_n - 1, idx_m] = la.norm(true_grad-v, 2)

                # compute time for GD with timeit
                #t = timeit.Timer((functools.partial(gradient_chol, a, b, M, reg, L_GD, tresh)))
                #time_detected = t.timeit(number=20) / 20
                #print(time_detected)

                # compute time with time.time
                t_init = time.time()
                grad = gradient_chol(a, b, M, reg, L_GD, tresh)
                t_end = time.time()
                time_detected = t_end - t_init
                gd_time[idx_seed, len(dim1) - idx_n - 1, idx_m] = time_detected

                tM = torch.DoubleTensor(M)
                tb = torch.DoubleTensor(b)
                #tM = torch.from_numpy(M).type(dtype=torch.double)
                #tb = torch.from_numpy(b).type(dtype=torch.double)

                v = grad_AD_double(a, tb, tM, reg, L_AD, tresh)[2]

                # accuracy for AD
                v = v.data.numpy()
                ad_score[idx_seed, len(dim1) - idx_n - 1, idx_m] = la.norm(true_grad - v, 2)

                # compute time for AD with timeit
                #t = timeit.Timer((functools.partial(grad_AD_double, a, tb, tM, reg, L_AD, tresh)))
                #time_detected = t.timeit(number=20) / 20
                #print(time_detected)
                
                t_init = time.time()
                grad = grad_AD_double(a, tb, tM, reg, L_AD, tresh)
                t_end = time.time()
                time_detected = t_end - t_init
                
                #AD_time[dim1.shape[0]-idx_n-1, idx_m] = t2 - t1
                ad_time[idx_seed, len(dim1) - idx_n - 1, idx_m] = time_detected

    create_my_plot(ad_time, gd_time)
    print('end')
