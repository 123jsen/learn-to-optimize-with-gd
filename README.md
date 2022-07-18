# learn-to-optimize-with-gd
Simple Project for implementing Learn to Optimize

## Project Scope
We will train a L2O optimizer by using it to train other optimizees. The cost function for the optimizer will be accumulated cost from the optimizees.

## Theory
Denote the parameters of the optimizee as $\theta$, that of the optimizer as $\phi$.

1. Training data is forward passed into the optimizee, which creates a prediction and loss $\ell$. 
2. Gradients $\frac{\partial \ell}{\partial \theta}$ for the optimizee are calculated.
3. The gradient $\frac{\partial \ell}{\partial \theta}$ is passed into the L2O optimizer, which outputs a step size to update the optimizee.
4. Steps 1 to 3 are repeated for $n$ iterations, which produces an accumulated loss $\mathcal{L}$.
5. Gradients $\frac{\partial \mathcal{C}}{\partial \phi}$ are calculated and used to update $\phi$ using traditional gradient descent.

## Implementation
We will train a regression model using the Boston Housing Prices dataset.
