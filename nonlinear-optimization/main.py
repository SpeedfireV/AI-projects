import time
from typing import NamedTuple, Callable, Sequence, List, Dict, Any
import numpy as np

class SolverParameters(NamedTuple):
    fitness: float = 0.05  # Liczba wywołań funkcji celu [0, 1] * FE_max
    sigma: float = 10  # Siła mutacji
    a: int = 1  # Interwał adaptacji sigma



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
    FE_max =  100 * len(x0) # FE_max
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
        print(params)
        history.append(params)
        print(len(history))
    stop = time.time()
    time_s = stop - start
    return SolverResult(x_best=x0, f_best=o_m, evals=t, history=params, params=params, time_s=time_s)


if __name__ == "__main__":
	def f(x):
		return x[0]**2 + x[1]**2 + 50
	x0 = np.array([1, 1])
	res = solver(f, x0, SolverParameters())
	print(res)