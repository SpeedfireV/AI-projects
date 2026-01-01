import numpy as np
from enum import Enum, auto
from math import sin, cos, pow
from scipy.optimize import minimize, differential_evolution

class Method(Enum):
    Evolutionary = auto()
    Gradient = auto()

class MLP:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.size = (n_input * n_hidden) + n_hidden + (n_hidden * n_output) + n_output

    def forward(self, x, weights):
        idx = 0
        w1 = weights[idx:idx + self.n_input * self.n_hidden].reshape(self.n_input, self.n_hidden)
        idx += self.n_input * self.n_hidden
        b1 = weights[idx:idx + self.n_hidden]
        idx += self.n_hidden
        w2 = weights[idx:idx + self.n_hidden * self.n_output].reshape(self.n_hidden, self.n_output)
        idx += self.n_hidden * self.n_output
        b2 = weights[idx:idx + self.n_output]

        z1 = np.dot(x, w1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, w2) + b2
        return z2

def base_function(x):
    return pow(x, 2) * sin(x) + 100 * sin(x) * cos(x)

def objective_function(weights, model, x_train, y_train):
    predictions = model.forward(x_train, weights)
    mse = np.mean((predictions.flatten() - y_train)**2)
    return mse

def solve(n_neurons, method_type):
    x_train = np.linspace(-10, 10, 1000).reshape(-1, 1)
    y_train = np.array([base_function(val[0]) for val in x_train])

    model = MLP(n_input=1, n_hidden=n_neurons, n_output=1)
    initial_weights = np.random.uniform(-1, 1, model.size)

    if method_type == Method.Gradient:
        # With L-BFGS-B it isn't required to set learning_rate as it learns by itself
        res = minimize(objective_function, initial_weights, 
                       args=(model, x_train, y_train), method='L-BFGS-B')
    else:
        # IMPORTANT - maxiter can be changed for evolutionary algorithm
        bounds = [(-10, 10)] * model.size
        res = differential_evolution(objective_function, bounds, 
                                     args=(model, x_train, y_train), maxiter=50)

    return res.fun, res.x, model, x_train, y_train


if __name__ == "__main__":
    neurons_to_test = [2, 10, 50]

    for n in neurons_to_test:
        mse_grad, weights_grad, model, x_t, y_t = solve(n, Method.Gradient)
        print(f"Neurons: {n} | Method: Gradient | MSE: {mse_grad:.4f}")
        mse_grad, weights_grad, model, x_t, y_t = solve(n, Method.Evolutionary)
        print(f"Neurons: {n} | Method: Evolutionary | MSE: {mse_grad:.4f}")