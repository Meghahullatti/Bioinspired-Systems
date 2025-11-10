import random
import multiprocessing as mp
import time

# ===========================================================
# ROAD CLASS
# ===========================================================
class Road:
    def __init__(self, length, density, max_speed):
        self.length = length
        self.max_speed = max_speed
        self.cells = [0] * length
        self.randomize(density)
    
    def randomize(self, density):
        n_cars = int(density * self.length)
        for i in range(n_cars):
            self.cells[i] = 1
        random.shuffle(self.cells)
    
    def step(self):
        new_cells = [0] * self.length
        for i in range(self.length):
            if self.cells[i] == 1:
                distance = 1
                while distance < self.max_speed and self.cells[(i + distance) % self.length] == 0:
                    distance += 1
                move = min(distance - 1, self.max_speed)
                new_pos = (i + move) % self.length
                new_cells[new_pos] = 1
        self.cells = new_cells

# ===========================================================
# TRAFFIC SIGNAL CONTROLLER (RED → YELLOW → GREEN)
# ===========================================================
class TrafficSignal:
    def __init__(self, red_time, yellow_time, green_time):
        self.red_time = red_time
        self.yellow_time = yellow_time
        self.green_time = green_time
        self.timer = 0
        self.state = "RED"
    
    def update(self, queue_length):
        self.timer += 1
        
        if self.state == "RED" and self.timer > self.red_time + 0.1 * queue_length:
            self.state = "YELLOW"
            self.timer = 0
        
        elif self.state == "YELLOW" and self.timer > self.yellow_time:
            self.state = "GREEN"
            self.timer = 0
        
        elif self.state == "GREEN" and self.timer > self.green_time:
            self.state = "RED"
            self.timer = 0

# ===========================================================
# PARALLEL UPDATE FUNCTION
# ===========================================================
def update_road(road):
    road.step()
    return road

# ===========================================================
# MAIN SIMULATION LOOP
# ===========================================================
def simulate():
    print("\n===== TRAFFIC FLOW SIMULATION (RED → YELLOW → GREEN) =====\n")
    
    # ---- USER INPUT ----
    num_roads = int(input("Enter number of roads: "))
    road_length = int(input("Enter road length (e.g., 50): "))
    car_density = float(input("Enter car density (0.1 - 0.9): "))
    max_speed = int(input("Enter maximum car speed (1-10): "))
    red_time = int(input("Enter red signal time: "))
    yellow_time = int(input("Enter yellow signal time: "))
    green_time = int(input("Enter green signal time: "))
    sim_steps = int(input("Enter number of simulation steps: "))
    time_step = float(input("Enter delay between steps (seconds): "))

    # ---- INITIALIZATION ----
    roads = [Road(road_length, car_density, max_speed) for _ in range(num_roads)]
    signal = TrafficSignal(red_time, yellow_time, green_time)
    
    pool = mp.Pool(mp.cpu_count())
    print("\nSimulation started...\n")

    for t in range(sim_steps):
        roads = pool.map(update_road, roads)

        total_cells = num_roads * road_length
        total_cars = sum(sum(r.cells) for r in roads)
        avg_density = total_cars / total_cells
        total_queue = sum(sum(r.cells[-5:]) for r in roads)

        signal.update(total_queue)

        if t % 5 == 0:
            print(f"Step {t:03d} | Density={avg_density:.3f} | Queue={total_queue:3d} | Signal={signal.state}")

        time.sleep(time_step)

    pool.close()
    pool.join()
    print("\nSimulation completed successfully!\n")

# ===========================================================
# RUN PROGRAM
# ===========================================================
if __name__ == "__main__":
    simulate()
