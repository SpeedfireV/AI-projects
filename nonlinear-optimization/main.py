from typing import NamedTuple, Callable, Sequence, List, Dict, Any
import numpy as np

class SolverParameters(NamedTuple):
	t_max: int = 20000 # Liczba iteracji
	sigma: float = 0.1  # Siła mutacji
	a: float = 1.0  # Interwał adaptacji sigma


class SolverResult(NamedTuple):
	x_best: np.ndarray
	f_best: float
	evals: int
	history: Dict[str, List[Any]]
	params: SolverParameters
	time_s: float



def solver(
		eval_func: Callable[[Sequence[float]], float], # Funkcja celu
		x0: Sequence[float], # Osobnik początkowy
		params: SolverParameters,
	) -> SolverResult:
	params._replace(t_max = int(params.t_max) * len(x0)) # FE_max

	t = 1
	l_s = 0
	o0 = eval_func(x0)
	while t <= params.t_max:
		z = np.random.uniform(-1, 1, len(x0)) # Rozkład normalny
		m = x0 + params.sigma * z
		o_m = eval_func(m) # Wartość dla mutanta
		if o_m <= o0:
			l_s += 1
			o0 = o_m
			x0 = m
		if t % params.a == 0:
			if l_s / params.a >= 1/5:
				params = params._replace(sigma = params.sigma * 1.22)
			if l_s / params.a <= 1/5:
				params = params._replace(sigma = params.sigma  * 0.82)
			l_s = 0
		t += 1

if __name__ == "__main__":
	def f(x):
		return x[0]**2 + x[1]**2
	x0 = np.array([1, 1])
	res = solver(f, x0, SolverParameters())
	print(res)