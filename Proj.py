import numpy as np
import random
import heapq

def create_rooms():
    rooms_data = {}
    neighbors = {}
    heuristic = {}
    num_rooms = int(input("Enter the number of rooms: "))

    for i in range(num_rooms):
        room_name = input(f"Enter name for room {i+1}: ")
        length = float(input(f"Enter length for room {room_name}: "))
        width = float(input(f"Enter width for room {room_name}: "))
        heuristic_value = int(input("Enter the heuristic of the room: "))
        rooms_data[room_name] = (length, width)
        heuristic[room_name] = heuristic_value

    for room in rooms_data:
        connected_rooms = input(f"Enter neighboring rooms for {room} (comma-separated): ").split(',')
        neighbors[room] = [neighbor.strip() for neighbor in connected_rooms]
    rooms = list(rooms_data.keys())

    return rooms_data, neighbors, rooms, heuristic

# Function to calculate the path cost
def get_path_cost(path):
    if len(rooms_data) > 0:
        return sum([np.linalg.norm(rooms_data[room]) for room in path])

# Function to find the optimal path using breadth-first search
def breadth_first_search(start, end):
    queue = [[start]]
    visited = set()

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == end:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in neighbors[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

# Function to find the optimal path using depth-first search
def depth_first_search(start, end):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == end:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in neighbors[node]:
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

# Function to find the optimal path using ant colony optimization
def ant_colony_optimization(start, end, rooms_data):
    
    room_indices = {room: i for i, room in enumerate(rooms_data.keys())}
    num_rooms = len(rooms_data)

    pheromone = np.ones((len(rooms_data), len(rooms_data)))
    iterations = 100

    for i in range(iterations):
        ant_paths = []

        for ant in range(50):
            current_room = start
            path = [current_room]
            
            while current_room != end:
                possible_neighbors = neighbors[current_room]
                probabilities = []

                for neighbor in possible_neighbors:
                    pheromone_level = pheromone[room_indices[current_room]][room_indices[neighbor]]
                    heuristic = 1 / (np.linalg.norm(rooms_data[current_room]) + np.linalg.norm(rooms_data[neighbor]))
                    probabilities.append((neighbor, pheromone_level * heuristic))

                total_prob = sum(p[1] for p in probabilities)
                probabilities = [(n, p / total_prob) for n, p in probabilities] if total_prob != 0 else probabilities

                next_room = np.random.choice([n for n, _ in probabilities], p=[p for _, p in probabilities])
                path.append(next_room)
                current_room = next_room

            ant_paths.append((path, get_path_cost(path)))

        for path, cost in ant_paths:
            for idx in range(len(path) - 1):
                room_a_idx, room_b_idx = rooms_data.index(path[idx]), rooms_data.index(path[idx + 1])
                pheromone[room_a_idx, room_b_idx] += 1 / cost

    best_path = max(ant_paths, key=lambda x: x[1])[0]
    return best_path

def hill_climbing(start, end, heuristic, neighbors):
    current_room = start
    path = []

    while True:
        path.append(current_room)
        if current_room == end:
            print(f"Hill Climbing: Reached the end room {end}, path taken: {path}")
            return

        next_room = min(neighbors, key=lambda x: heuristic[x], default=None)
        
        if next_room is None or heuristic[next_room] >= heuristic[current_room]:
            print(f"Hill Climbing: Local maximum reached at {current_room}, path taken: {path}")
            return
        current_room = next_room

def particle_swarm_optimization(start, end, rooms_data, neighbors):
    num_particles = 50
    num_iterations = 100
    inertia_weight = 0.7
    cognitive_param = 1.4
    social_param = 1.4

    def initialize_particle():
        return {
            "position": [random.randint(0, len(neighbors[room]) - 1) for room in neighbors],
            "velocity": [0 for _ in range(len(neighbors))],
            "personal_best": None,
            "personal_best_cost": float('inf'),
            "global_best": None,
            "global_best_cost": float('inf')
        }

    particles = [initialize_particle() for _ in range(num_particles)]

    for _ in range(num_iterations):
        for particle in particles:
            # Convert indices to room names
            position_rooms = [neighbors[room][idx] for room, idx in zip(neighbors, particle["position"])]
            cost = get_path_cost(position_rooms)

            if cost < particle["personal_best_cost"]:
                particle["personal_best"] = list(particle["position"])
                particle["personal_best_cost"] = cost

            if cost < particle["global_best_cost"]:
                particle["global_best"] = list(particle["position"])
                particle["global_best_cost"] = cost

        for particle in particles:
            for i in range(len(particle["position"])):
                r1, r2 = random.random(), random.random()
                cognitive = cognitive_param * r1 * (particle["personal_best"][i] - particle["position"][i])
                social = social_param * r2 * (particle["global_best"][i] - particle["position"][i])

                particle["velocity"][i] = inertia_weight * particle["velocity"][i] + cognitive + social

                new_position = max(min(particle["position"][i] + particle["velocity"][i], len(neighbors[rooms[i]]) - 1), 0)
                particle["position"][i] = int(new_position)

    best_particle = min(particles, key=lambda p: get_path_cost([neighbors[room][idx] for room, idx in zip(neighbors, p["position"])]))
    best_path = [neighbors[room][idx] for room, idx in zip(neighbors, best_particle["position"])]
    return best_path


def astar(start_room, end_room, heuristic, neighbors):
    

    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (heuristic, start_room, []))

    while open_set:
        _, room, path = heapq.heappop(open_set)

        if room == end_room:
            print(f"Reached destination {room}, path taken: {path + [room]}")
            return path + [room]

        if room not in closed_set:
            closed_set.add(room)

            for next_room in neighbors:
                if next_room not in closed_set:
                    heapq.heappush(open_set, (heuristic[next_room], next_room, path + [room]))

    return []

def select_algorithm(start, end, heuristic, rooms, neighbors, rooms_data):
    while True:
        algorithm_choice = input("Enter the algorithm you want to use: (BFS, DFS, HC, GA, PSO, A*, or QUIT): ")

        if algorithm_choice.upper() == "QUIT":
            break
        elif algorithm_choice.upper() == "BFS":
            path = breadth_first_search(start, end)
        elif algorithm_choice.upper() == "DFS":
            path = depth_first_search(start, end)
        elif algorithm_choice.upper() == "HC":
            path = hill_climbing(start, end, heuristic, neighbors)
        elif algorithm_choice.upper() == "PSO":
            path = particle_swarm_optimization(start, end, rooms_data, neighbors)
        elif algorithm_choice.upper() == "A*":
            path = astar(start, end, heuristic, neighbors)
        else:
            print("Invalid algorithm choice. Please enter a valid option.")
            continue
        
        if path:
            path_cost = get_path_cost(path)
            print("Optimal Path:", path)
            print("Optimal Path Cost:", path_cost)
        else:
            print("No path found using the selected algorithm.")

rooms_data, neighbors, rooms, heuristic = create_rooms()
start_room = input("Enter the start room: ")
end_room = input("Enter the end room: ")
select_algorithm(start_room, end_room, heuristic, rooms, neighbors, rooms_data)
