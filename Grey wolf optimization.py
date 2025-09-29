import numpy as np

# Objective function: f(x) = x^2
def objective(x):
    return x[0] ** 2

# Grey Wolf Optimizer (1D example)
def gwo(obj_function, dim, search_agents_no, max_iter, lb, ub):
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    # Initialize search agents randomly
    positions = np.random.uniform(lb, ub, (search_agents_no, dim))

    for l in range(max_iter):
        for i in range(search_agents_no):
            # Keep agents within bounds
            positions[i] = np.clip(positions[i], lb, ub)

            # Fitness
            fitness = obj_function(positions[i])

            # Update alpha, beta, delta
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i].copy()

        # Parameter 'a' decreases linearly
        a = 2 - l * (2 / max_iter)

        for i in range(search_agents_no):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i][j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score

# Run GWO for f(x) = x^2
best_pos, best_score = gwo(objective, dim=1, search_agents_no=5, max_iter=30, lb=-10, ub=10)

print("Best Position (x):", best_pos)
print("Best Score f(x):", best_score)

