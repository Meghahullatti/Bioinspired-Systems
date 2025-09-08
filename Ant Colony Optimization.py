import random
import math

# Generate random cities as (x, y) tuples
def generate_cities(n, seed=42):
    random.seed(seed)
    return [(random.random(), random.random()) for _ in range(n)]

# Euclidean distance between two points
def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# Create distance matrix
def create_distance_matrix(cities):
    n = len(cities)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = distance(cities[i], cities[j])
    return dist_matrix

class AntColony:
    def __init__(self, dist_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        self.dist_matrix = dist_matrix
        self.pheromone = [[1 / len(dist_matrix) for _ in range(len(dist_matrix))] for _ in range(len(dist_matrix))]
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_cities = len(dist_matrix)

    def run(self):
        all_time_shortest_path = (None, float('inf'))

        for iteration in range(self.n_iterations):
            all_paths = self.construct_paths()
            self.spread_pheromone(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.evaporate_pheromone()
            print(f"Iteration {iteration+1}/{self.n_iterations}, best distance: {all_time_shortest_path[1]:.4f}")

        return all_time_shortest_path

    def construct_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.construct_path(0)
            distance = self.path_distance(path)
            all_paths.append((path, distance))
        return all_paths

    def construct_path(self, start):
        path = [start]
        visited = set(path)

        for _ in range(self.n_cities - 1):
            current_city = path[-1]
            next_city = self.pick_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)

        path.append(start)  # Return to start
        return path

    def pick_next_city(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        dist = self.dist_matrix[current_city]

        probabilities = []
        denominator = 0

        for city in range(self.n_cities):
            if city not in visited:
                pher = pheromone[city] ** self.alpha
                heuristic = (1 / dist[city]) ** self.beta if dist[city] > 0 else 0
                prob = pher * heuristic
                probabilities.append((city, prob))
                denominator += prob

        if denominator == 0:
            # Choose random unvisited city
            choices = [city for city in range(self.n_cities) if city not in visited]
            return random.choice(choices)

        # Roulette wheel selection
        r = random.uniform(0, denominator)
        total = 0
        for city, prob in probabilities:
            total += prob
            if total >= r:
                return city

    def path_distance(self, path):
        total = 0
        for i in range(len(path) - 1):
            total += self.dist_matrix[path[i]][path[i + 1]]
        return total

    def spread_pheromone(self, all_paths, n_best):
        all_paths.sort(key=lambda x: x[1])
        for path, dist in all_paths[:n_best]:
            for i in range(len(path) - 1):
                from_city = path[i]
                to_city = path[i + 1]
                # Deposit pheromone inversely proportional to distance
                self.pheromone[from_city][to_city] += 1.0 / dist
                self.pheromone[to_city][from_city] += 1.0 / dist  # symmetric

    def evaporate_pheromone(self):
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.pheromone[i][j] *= (1 - self.decay)
                # Avoid pheromone dropping to zero
                if self.pheromone[i][j] < 0.0001:
                    self.pheromone[i][j] = 0.0001

def main():
    n_cities = 10
    cities = generate_cities(n_cities)
    dist_matrix = create_distance_matrix(cities)

    colony = AntColony(
        dist_matrix,
        n_ants=20,
        n_best=5,
        n_iterations=10,
        decay=0.1,
        alpha=1,
        beta=5
    )

    best_path, best_dist = colony.run()
    print("\nBest path:", best_path)
    print(f"Best distance: {best_dist:.4f}")

if __name__ == "__main__":
    main()
