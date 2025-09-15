import numpy as np
from scipy.special import gamma

# Parameters for the Cuckoo Search
n = 20             # Number of nests (i.e., population size)
max_iter = 100     # Maximum number of iterations
pa = 0.25          # Probability of an egg being discovered
alpha = 1.0        # Step size
beta = 1.5         # Levy flight parameter
W = 50             # Capacity of the knapsack (total weight limit)

# Define the items: weights and values
weights = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
values = np.array([60, 100, 120, 160, 200, 240, 280, 300, 350, 400])
num_items = len(weights)

# Objective function: Calculate the total value of selected items
# Apply a heavy penalty for invalid solutions (weight exceeds capacity)
def fitness_function(solution):
    total_weight = np.sum(weights * solution)
    total_value = np.sum(values * solution)
   
    if total_weight > W:
        # Severe penalty for exceeding the weight capacity
        return 0  # This will discard invalid solutions
   
    return total_value

# Levy flight function for generating new solutions
def levy_flight(beta, size):
    sigma_u = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / gamma((1 + beta) / 2) * np.power(gamma(beta / 2), 2), 1 / beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

# Initialize nests (solutions) - each nest is a binary vector representing selected items
def initialize_nests(n, num_items):
    nests = np.random.randint(0, 2, (n, num_items))  # Random binary vector for each nest
    return nests

# Main Cuckoo Search algorithm for Knapsack
def cuckoo_search_knapsack(n, max_iter, num_items, weights, values, W, pa, alpha, beta):
    nests = initialize_nests(n, num_items)
    fitness = np.array([fitness_function(nests[i]) for i in range(n)])
    best_nest = nests[np.argmax(fitness)]  # Find the nest with the highest fitness (max value)
    best_fitness = np.max(fitness)
   
    t = 0  # Initialize iteration counter

    while t < max_iter:
        # Generate new solutions (cuckoos) using Levy flight
        new_nests = nests + alpha * levy_flight(beta, (n, num_items))
        new_nests = np.clip(new_nests, 0, 1)  # Ensure binary solutions (0 or 1)
       
        # Evaluate new solutions
        new_fitness = np.array([fitness_function(new_nests[i]) for i in range(n)])
       
        # Replace nests with better solutions
        for i in range(n):
            if new_fitness[i] > fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
       
        # Find the best nest
        best_nest_idx = np.argmax(fitness)
        best_fitness = fitness[best_nest_idx]
        best_nest = nests[best_nest_idx]
       
        # Randomly abandon a fraction Pa of the worst nests
        sorted_idx = np.argsort(fitness)
        worst_nests = sorted_idx[:int(pa * n)]
        nests[worst_nests] = np.random.randint(0, 2, (len(worst_nests), num_items))  # Replace with random solutions
        fitness[worst_nests] = np.array([fitness_function(nests[i]) for i in worst_nests])

        # Print progress
        if t % 10 == 0:
            print(f"Iteration {t}, Best Fitness (Total Value): {best_fitness}")
       
        t += 1
   
    return best_nest, best_fitness

# Running the cuckoo search for Knapsack Problem
best_solution, best_value = cuckoo_search_knapsack(n, max_iter, num_items, weights, values, W, pa, alpha, beta)

print("\nBest Solution (Binary Representation): ", best_solution)
print("Best Fitness Value (Total Value): ", best_value)

# Decoding the solution to display selected items
selected_items = np.where(best_solution == 1)[0]
print("Selected Items (Indices): ", selected_items)
print("Total Weight of Selected Items: ", np.sum(weights[selected_items]))

