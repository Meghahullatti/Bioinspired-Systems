import numpy as np

# Objective function to minimize
def f(x, y):
    return x**2 + y**2

# Parameterss
grid_size = 10
alpha = 0.1           # learning rate
iterations = 100
cells = np.random.uniform(-5, 5, (grid_size, grid_size, 2))  # (x, y) per cell

for _ in range(iterations):
    # Evaluate fitness for all cells
    fitness = f(cells[:, :, 0], cells[:, :, 1])
   
    new_cells = np.copy(cells)
    for i in range(grid_size):
        for j in range(grid_size):
            # Get neighbors (Moore neighborhood)
            ni = [max(i-1,0), i, min(i+1,grid_size-1)]
            nj = [max(j-1,0), j, min(j+1,grid_size-1)]
           
            neighbors = np.array([cells[a,b] for a in ni for b in nj])
            fit_vals = np.array([f(n[0], n[1]) for n in neighbors])
           
            best_neighbor = neighbors[np.argmin(fit_vals)]
           
            # Update rule
            new_cells[i,j] = cells[i,j] + alpha * (best_neighbor - cells[i,j]) \
                             + np.random.uniform(-0.05, 0.05, 2)
   
    cells = new_cells

# Find best solution
fitness = f(cells[:,:,0], cells[:,:,1])
best_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
best_x, best_y = cells[best_idx]
best_f = fitness[best_idx]

print(f"Best solution: x={best_x:.4f}, y={best_y:.4f}, f(x,y)={best_f:.6f}")
