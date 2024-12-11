import numpy as np
import matplotlib.pyplot as plt


# Function and Gradient
def f(x, y):
    return x ** 2 + 20 * y ** 2


def grad_f(x, y):
    return np.array([2 * x, 40 * y])


# Optimization Methods
def optimize(optimizer, x_init, y_init, alpha, steps, **kwargs):
    x, y = x_init, y_init
    trajectory = [(x, y)]  # To track x, y over time

    # Initialize variables for momentum and Adam
    vx, vy = 0, 0
    mx, my, vx_adam, vy_adam = 0, 0, 0, 0
    beta1, beta2, epsilon = kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999), 1e-8

    for t in range(1, steps + 1):
        grad = grad_f(x, y)

        if optimizer == "sgd":
            # Standard Gradient Descent
            x -= alpha * grad[0]
            y -= alpha * grad[1]

        elif optimizer == "momentum":
            # Gradient Descent with Momentum
            rho = kwargs.get('rho', 0.9)
            vx = rho * vx - alpha * grad[0]
            vy = rho * vy - alpha * grad[1]
            x += vx
            y += vy

        elif optimizer == "adam":
            # Adam Optimizer
            mx = beta1 * mx + (1 - beta1) * grad[0]
            my = beta1 * my + (1 - beta1) * grad[1]
            vx_adam = beta2 * vx_adam + (1 - beta2) * (grad[0] ** 2)
            vy_adam = beta2 * vy_adam + (1 - beta2) * (grad[1] ** 2)

            m_x_hat = mx / (1 - beta1 ** t)
            m_y_hat = my / (1 - beta1 ** t)
            v_x_hat = vx_adam / (1 - beta2 ** t)
            v_y_hat = vy_adam / (1 - beta2 ** t)

            x -= alpha * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
            y -= alpha * m_y_hat / (np.sqrt(v_y_hat) + epsilon)

        trajectory.append((x, y))

    return np.array(trajectory)


# Plot results
def plot_trajectory(trajectories, labels):
    plt.figure(figsize=(10, 5))
    for trajectory, label in zip(trajectories, labels):
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory of Optimization")
    plt.legend()
    plt.show()


# Initial Parameters
x_init, y_init = -20, 5
alpha = 0.1
steps = 200

# Run Optimizations
trajectory_sgd = optimize("sgd", x_init, y_init, alpha, steps)
trajectory_momentum = optimize("momentum", x_init, y_init, alpha, steps, rho=0.9)
trajectory_adam = optimize("adam", x_init, y_init, alpha, steps, beta1=0.9, beta2=0.999)

# Plot Results
plot_trajectory(
    [trajectory_sgd, trajectory_momentum, trajectory_adam],
    ["SGD", "Momentum", "Adam"]
)
