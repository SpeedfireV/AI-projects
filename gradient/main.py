import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def project_to_domain(x, low=-10.0, high=10.0):
    return np.clip(x, low, high)

def backtracking(f, grad_f, x, step, alpha=0.3, beta=0.7):
    f_x = f(x)
    g = grad_f(x)

    while True:
        x_new = project_to_domain(x - step * g)
        if f(x_new) <= f_x - alpha * step * np.dot(g, g):
            return x_new, step
        step *= beta

def gradient_descent(
        f, grad_f, x0,
        alpha=1.0,
        max_iter=5000,
        domain_low=-10.0,
        domain_high=10.0,
        use_backtracking=True,
        max_step_norm=1.0):

    x = project_to_domain(np.array(x0, dtype=float), domain_low, domain_high)

    f_hist = []
    grad_hist = []

    for t in range(max_iter):

        g = grad_f(x)
        grad_norm = np.linalg.norm(g)

        f_hist.append(f(x))
        grad_hist.append(grad_norm)

        if grad_norm < 1e-12:
            break

        step = alpha

        if use_backtracking:
            x, step_used = backtracking(f, grad_f, x, step)
        else:
            step_vec = -step * g
            if np.linalg.norm(step_vec) > max_step_norm:
                step_vec = step_vec / np.linalg.norm(step_vec) * max_step_norm
            x = project_to_domain(x + step_vec)

    return np.array(f_hist), np.array(grad_hist)


def quadratic(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = x.size
    return (-a * np.exp(-b*np.sqrt(np.sum(x**2)/d))
            - np.exp(np.sum(np.cos(c*x))/d)
            + a + np.e)

def run_experiment(f, f_name, n=10):
    grad_f = grad(f)

    alphas = [1, 10, 100]
    x0 = np.random.uniform(-10, 10, size=n)

    histories_f = {}
    histories_g = {}

    print(f"--- Uruchamiam eksperyment dla: {f_name} ---")

    for alpha in alphas:
        f_hist, g_hist = gradient_descent(f, grad_f, x0, alpha=alpha)
        histories_f[alpha] = f_hist
        histories_g[alpha] = g_hist

    fig, ax = plt.subplots(1, 2, figsize=(14,5))

    for alpha in alphas:
        ax[0].plot(histories_f[alpha], label=f"α={alpha}")
    ax[0].set_title(f"{f_name}: f(x_t)")
    ax[0].set_xlabel("iteracje")
    ax[0].set_ylabel("f")
    ax[0].legend()

    for alpha in alphas:
        ax[1].plot(histories_g[alpha], label=f"α={alpha}")
    ax[1].set_title(f"{f_name}: ||grad||")
    ax[1].set_xlabel("iteracje")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("log ||grad||")
    ax[1].legend()

    filename = f"{f_name}_results.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Zapisano wykres do pliku: {filename}")

if __name__ == "__main__":
    run_experiment(quadratic, "Quadratic")
    run_experiment(rosenbrock, "Rosenbrock")
    run_experiment(ackley, "Ackley")