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
We will implement L2O step by step:

1. Regression without L2O
2. Simple Regression on generated data using a gradient learning rate learnt by L2O
3. Learning rate L2O training on a Neural Network model
4. LSTM L2O Model training Neural Network and other classification problems. The LSTM model is applied pointwise to every parameter value (not tensor, parameter).

## Pseudocode for LSTM meta-training

    While (count < num_optimizee):
        Generate Hidden State
        Initialize Array of Optimizees
        While (epoch < num_epoch):
            While (batch is not end):
                Feedforward Next Batch through Optimizee
                if (iter == unroll_length):
                    Backpropagation on Optimizer
                    Reset Computational Graph
                    Reset Total Loss
                else:
                    Backpropagation on Optimizee


See "Training Stronger Baselines for Learning to Optimize" by Chen et al. for more.