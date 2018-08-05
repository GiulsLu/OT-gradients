# OT-gradients

This repository provides an easy tool for testing computational time and accuracy of gradients of Sinkhorn distances [1] computed with two different methods. 

One method leverages the analytic form of the gradient [4], [5], obtained through the Implicit Function Theorem; the linear system is solved using Cholesky factorization.

The other method uses Automatic Differentiation of Sinkhorn algorithm [3] : we included the function for convenience, to provide an 'easy-to-play' timing test but we do not claim _any novelty_ in the algorithm. 


## Getting Started


### Prerequisites

To run the codes and use the functions in this respository you will need to install Python Optimal Transport Library first (https://github.com/rflamary/POT).


## Running the tests
- File run_regimes.py contains a code that automatically runs time and accuracy experiments with pre-set choices of parameters and outputs plots with the performance. Pre-set parameters are the following:
  n = dimension of one histogram
  m = dimension of the other histogram
  nr_seeds = number of different seeds on which the experiment is run
  iter_AD = number of Iteration of Sinkhorn algorithm when computing the gradient with automatic differentiation 
  reg = regularization paramenter.
  
  

- File regimes.py is version with command-line interfaces: parameters can be passed in when running the program; e.g.:

python regimes.py -n 200 1000 5000 -m 200 500 1000 -nr_seeds 3 -iter_AD 30 -reg 0.02

if an argument is missing, the default is used.

- File gradients.py contains functions that compute the gradient with  closed formula and automatic differentiation. 



## Examples
a,b are 1D histograms\\
M is the ground cost matrix
reg is the regularization
numIter and tresh correspond to number of iterations and treshold on the error for Sinkhorn Algorithm

>grad = gradient_chol(a,b,M,reg,numIter,tresh)

## References 
[1] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems (pp. 2292-2300).

[2] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

[3] Genevay, A., Peyré, G., Cuturi, M.(2018): Learning Generative Models with Sinkhorn Divergences, Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics.

[4] Luise, G., Rudi, A., Pontil, M., Ciliberto, C.(2018): Differential Properties of Sinkhorn Approximation for Learning with Wasserstein Distance, 	arXiv:1805.11897.

[5] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016). Wasserstein Discriminant Analysis. arXiv preprint arXiv:1608.08063.
