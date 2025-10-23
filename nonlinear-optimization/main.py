import random
import time
from typing import NamedTuple, Callable, Sequence, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class SolverParameters(NamedTuple):
    fitness: float = 0.05
    sigma: float = 0.3
    a: int = 10
    t_max: int = 100
    seed: Optional[int] = None

class HistoryElement(NamedTuple):
    x: Sequence[float]
    sigma: float
    f: float
    t: int

class SolverResult(NamedTuple):
    x_best: np.ndarray
    f_best: float
    evals: int
    history: List[HistoryElement]
    params: SolverParameters
    time_s: float

def solver(eval_func: Callable[[Sequence[float]], float],
           x0: Sequence[float],
           params: SolverParameters) -> SolverResult:

    if params.seed is not None:
        np.random.seed(params.seed)
        random.seed(params.seed)

    start = time.time()
    FE_max = params.t_max * len(x0)
    params = params._replace(fitness=params.fitness * FE_max)

    t = 1
    l_s = 0
    o0 = eval_func(x0)
    last_improvement = t
    history: List[HistoryElement] = [HistoryElement(x=x0, sigma=params.sigma, f=o0, t=t)]

    while t <= FE_max and t - last_improvement <= params.fitness:
        z = np.random.uniform(-1, 1, len(x0))
        m = x0 + params.sigma * z
        o_m = eval_func(m)
        if o_m <= o0:
            l_s += 1
            o0 = o_m
            x0 = m
            last_improvement = t
        if t % params.a == 0:
            success_rate = l_s / params.a
            if success_rate >= 1/5:
                params = params._replace(sigma=params.sigma * 1.22)
            elif success_rate <= 1/5:
                params = params._replace(sigma=params.sigma * 0.82)
            l_s = 0
        t += 1
        history.append(HistoryElement(x=x0, sigma=params.sigma, f=o0, t=t))
    stop = time.time()
    return SolverResult(x_best=x0, f_best=o0, evals=t, history=history,
                        params=params, time_s=stop - start)

def quadratic(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray, a=1.0, b=100.0) -> float:
    x = np.asarray(x)
    return float(np.sum(b*(x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2))

def ackley(x: np.ndarray, a=20, b=0.2, c=2*np.pi) -> float:
    x = np.asarray(x)
    n = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c*x))
    return float(-a*np.exp(-b*np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + a + np.e)

def plot_multiple_histories(results: List[SolverResult], title: str, filename_base: str):
    plt.figure(figsize=(8, 5))
    all_histories = []
    all_sigmas = []

    for res in results:
        f_values = [h.f for h in res.history]
        iters = [h.t for h in res.history]
        sigma_values = [h.sigma for h in res.history]
        plt.plot(iters, f_values, color="lightgray", linewidth=1, alpha=0.6)
        all_histories.append(np.interp(np.linspace(1, max(iters), 200),
                                       iters, f_values))
        all_sigmas.append(np.interp(np.linspace(1, max(iters), 200),
                                    iters, sigma_values))

    all_histories = np.array(all_histories)
    median_curve = np.median(all_histories, axis=0)
    all_sigmas = np.array(all_sigmas)
    median_sigma = np.median(all_sigmas, axis=0)
    xs = np.linspace(1, max(iters), 200)

    os.makedirs("charts", exist_ok=True)
    plt.figure(figsize=(8, 5))
    for res in results:
        f_values = [h.f for h in res.history]
        iters = [h.t for h in res.history]
        plt.semilogy(iters, f_values, color="lightgray", linewidth=1, alpha=0.6)

    plt.semilogy(xs, median_curve, color="tab:blue", linewidth=2.5, label="Median")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("charts", f"{filename_base}.png"), dpi=200)
    plt.close()

    # Sigma(t) plot
    plt.figure(figsize=(8, 5))
    for res in results:
        sigma_values = [h.sigma for h in res.history]
        iters = [h.t for h in res.history]
        plt.plot(iters, sigma_values, color="lightgray", linewidth=1, alpha=0.6)
    plt.plot(xs, median_sigma, color="tab:orange", linewidth=2.5, label="Median sigma")
    plt.xlabel("Iteration")
    plt.ylabel("sigma")
    plt.title(f"Sigma(t) - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("charts", f"{filename_base}_sigma.png"), dpi=200)
    plt.close()

    # Zwracamy mediany do dalszego wykorzystania
    return xs, median_curve, median_sigma


def plot_all_medians(medians_dict, xs, title, filename_base, ylabel, legend_labels):
    plt.figure(figsize=(8, 5))
    for key, median_curve in medians_dict.items():
        plt.semilogy(xs, median_curve, label=legend_labels[key])
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs("charts", exist_ok=True)
    plt.savefig(os.path.join("charts", f"{filename_base}.png"), dpi=200)
    plt.close()


def run_experiments(eval_func, name: str, n_dim: int,
                    sigma_grid, a_grid, num_runs: int = 10):
    results_summary = []
    medians_f = {}
    medians_sigma = {}
    xs_ref = None
    legend_labels = {}
    for sigma in sigma_grid:
        for a in a_grid:
            key = f"{name}_n{n_dim}_sigma{sigma}_a{a}"
            print(f"\n=== Running {key} ===")
            results = []
            rows = []
            for i in range(num_runs):
                seed = 1000 + i
                np.random.seed(seed)
                x0 = np.random.normal(0, 5, n_dim)
                params = SolverParameters(sigma=sigma, a=a, t_max=100, seed=seed)
                res = solver(eval_func, x0, params)
                results.append(res)
                rows.append({
                    "function": name,
                    "n_dim": n_dim,
                    "sigma": sigma,
                    "a": a,
                    "run": i,
                    "seed": seed,
                    "f_best": res.f_best,
                    "evals": res.evals,
                    "time_s": res.time_s
                })
            xs, median_curve, median_sigma = plot_multiple_histories(results, key, key)
            if xs_ref is None:
                xs_ref = xs
            medians_f[(sigma, a)] = median_curve
            medians_sigma[(sigma, a)] = median_sigma
            legend_labels[(sigma, a)] = f"sigma={sigma}, a={a}"
            df_local = pd.DataFrame(rows)
            results_summary.append(df_local)
    return pd.concat(results_summary, ignore_index=True), medians_f, medians_sigma, xs_ref, legend_labels

if __name__ == "__main__":
    sigma_grid = [0.1, 1.0, 10.0]
    a_grid = [10, 100]
    num_runs = 10

    all_results = []

    for func, name in [(quadratic, "Quadratic"),
                       (rosenbrock, "Rosenbrock"),
                       (ackley, "Ackley")]:
        for n_dim in [10, 30]:
            df, medians_f, medians_sigma, xs, legend_labels = run_experiments(func, name, n_dim, sigma_grid, a_grid, num_runs)
            all_results.append(df)
            # Wykres zbiorczy median f(t)
            plot_all_medians(medians_f, xs, f"{name} n={n_dim} - Median f(x)", f"{name}_n{n_dim}_all_medians", "f(x)", legend_labels)
            # Wykres zbiorczy median sigma(t)
            plot_all_medians(medians_sigma, xs, f"{name} n={n_dim} - Median sigma(t)", f"{name}_n{n_dim}_all_sigmas", "sigma", legend_labels)

    df_all = pd.concat(all_results, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    df_all.to_csv("results/results.csv", index=False)
