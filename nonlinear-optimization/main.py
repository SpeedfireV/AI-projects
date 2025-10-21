import time
from typing import NamedTuple, Callable, Sequence, List, Dict, Any
import numpy as np

class SolverParameters(NamedTuple):
    fitness: float = 0.05  # Liczba wywołań funkcji celu [0, 1] * FE_max
    sigma: float = 0.3  # Siła mutacji
    a: int = 10  # Interwał adaptacji sigma
    t_max: int = 10000  # Maksymalna liczba iteracji

class SolverResult(NamedTuple):
    x_best: np.ndarray
    f_best: float
    evals: int
    history: List[SolverParameters]
    params: SolverParameters
    time_s: float

def solver(
        eval_func: Callable[[Sequence[float]], float], # Funkcja celu
        x0: Sequence[float], # Osobnik początkowy
        params: SolverParameters) -> SolverResult:
    start = time.time()
    print(len(x0))
    FE_max =  params.t_max * len(x0) # FE_max
    params = params._replace(fitness=params.fitness * FE_max)
    print(params.fitness)
    t = 1
    l_s = 0
    o0 = eval_func(x0)
    last_changed = t
    history: List[SolverParameters] = [params]
    while t <= FE_max and t - last_changed <= params.fitness:
        z = np.random.uniform(-1, 1, len(x0)) # Rozkład normalny
        m = x0 + params.sigma * z
        # print(m)
        o_m = eval_func(m) # Wartość dla mutanta
        if o_m <= o0:
            l_s += 1
            o0 = o_m
            x0 = m
        if t % params.a == 0:
            last_changed = t
            if l_s / params.a >= 1/5:
                params = params._replace(sigma = params.sigma * 1.22)
            if l_s / params.a <= 1/5:
                params = params._replace(sigma = params.sigma  * 0.82)
            l_s = 0
        t += 1
        history.append(params)
    stop = time.time()
    time_s = stop - start
    return SolverResult(x_best=x0, f_best=o_m, evals=t, history=params, params=params, time_s=time_s)

def quadratic(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray, a=1.0, b=100.0) -> float:
    # classic Rosenbrock for dim>=2
    x = np.asarray(x)
    return float(np.sum(b*(x[1:]-x[:-1]**2)**2 + (a-x[:-1])**2))

def ackley(x: np.ndarray, a=20, b=0.2, c=2*np.pi) -> float:
    x = np.asarray(x)
    n = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c*x))
    return float(-a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e)

if __name__ == "__main__":
    def f(x):
        return x[0]**2 + x[1]**2 + 50
    x0 = np.random.normal(0, 5, 2)
    res = solver(quadratic, x0, SolverParameters())
    print(res)
    x0 =  np.random.normal(-3, 3, 10)
    res = solver(rosenbrock, x0, SolverParameters())
    print(res)
    x0 = np.random.normal(-32, 32, 30)
    res = solver(ackley, x0, SolverParameters())
    print(res)