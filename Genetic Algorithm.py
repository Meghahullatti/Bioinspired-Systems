import random, math

# City coordinates
cities = {
    'New York': (0, 0), 'Boston': (2, 3), 'Philadelphia': (1, -1), 'Washington': (2, -3),
    'Chicago': (-3, 1), 'Detroit': (-2, 3), 'Miami': (4, -5), 'Atlanta': (3, -2)
}
city_names = list(cities.keys())
N = len(city_names)

# Parameters
POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.2

# Create random route
def create_route(): return random.sample(range(N), N)

# Calculate total distance of route
def route_distance(route):
    dist = 0
    for i in range(N):
        x1, y1 = cities[city_names[route[i - 1]]]
        x2, y2 = cities[city_names[route[i]]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

# Selection (Tournament)
def select(pop):
    return min(random.sample(pop, 3), key=route_distance)

# Crossover (Ordered)
def crossover(p1, p2):
    a, b = sorted(random.sample(range(N), 2))
    child = [-1]*N
    child[a:b] = p1[a:b]
    fill = [c for c in p2 if c not in child]
    idx = 0
    for i in range(N):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child

# Mutation (Swap)
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Main GA loop
def tsp_ga():
    pop = [create_route() for _ in range(POP_SIZE)]
    best = min(pop, key=route_distance)

    for gen in range(GENERATIONS):
        new_pop = [best]
        while len(new_pop) < POP_SIZE:
            p1, p2 = select(pop), select(pop)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)

        pop = new_pop
        current_best = min(pop, key=route_distance)
        if route_distance(current_best) < route_distance(best):
            best = current_best
        if gen % 10 == 0:
            print(f"Gen {gen}: {route_distance(best):.2f}")

    # Final output
    best_route_names = [city_names[i] for i in best]
    print("\nBest Route:", " → ".join(best_route_names + [best_route_names[0]]))
    print(f"Total Distance: {route_distance(best):.2f}")

# Run
if __name__ == "__main__":
    tsp_ga()
