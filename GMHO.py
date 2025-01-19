
"""
Script implements the Great Migration Herd Multi-Objective Optimization on objective functions defined in separate modules. 
Variable number of objective functions can be defined as per use cases..
"""

import numpy as np
import heapq

# Define your random objective functions here (replace with actual functions)
def objective_function_1(x, y, z):
    return x + y + z + np.random.rand()  # Adding random noise for demonstration

def objective_function_2(x, y, z):
    return x * y * z + np.random.rand()  # Adding random noise for demonstration

# Algorithm 1: Great Migration Herd Optimization
def great_migration_herd_optimization(num_objectives, objective_functions, relevance, total_particles, initial_solutions, solution_space):
    heap = []  # Initialize heap data structure

    for x in range(1, num_objectives + 1):
        # Select number of particles in each species
        P_x_min = 0  # Replace with your specific values
        P_x_max = 100  # Replace with your specific values
        P_N_x = np.random.randint(P_x_min, P_x_max + 1)

        # Randomly assign initial positions for particles
        initial_positions = np.random.normal(loc=initial_solutions[x-1], scale=1, size=(P_N_x, 3))

        # Initialize Heap data structure with data parts (Solution, Position)
        heap_data = [(objective_functions[x-1](*pos), pos) for pos in initial_positions]
        heapq.heapify(heap_data)
        heap.append(heap_data)

    destination_reached = False  # Replace with your termination condition
    t = 0
    # Define the maximum number of iterations
    max_iterations = 20

    while not destination_reached and t < max_iterations:
        t += 1
        for x in range(1, num_objectives + 1):
            # Algorithm 2: Movement of particles
            for i in range(len(heap[x-1])):
                position = heap[x-1][i][1]
                f_t = objective_functions[x-1](*position)
                f_t_minus_1 = objective_functions[x-1](*solution_space)
                a = 0.5  # Replace with your specific value
                b = 0.3  # Replace with your specific value
                f_t_plus_1_m = a * f_t - b * f_t_minus_1
                f_t_plus_1 = f_t_plus_1_m
                if abs(f_t_plus_1_m - f_t) < abs(f_t_plus_1 - f_t):
                    f_t_plus_1 = f_t_plus_1_m
                heap[x-1][i] = (f_t_plus_1, position)

            # Algorithm 3: Migration of species
            T_x = relevance[x-1]  # Set threshold based on significance relevance_x
            for i in range(len(heap[x-1])):
                for j in range(len(heap[x-1])):
                    if i != j:
                        d_ij = np.linalg.norm(np.array(heap[x-1][i][1]) - np.array(heap[x-1][j][1]))
                        if d_ij >= T_x:
                            theta = np.degrees(np.arccos(np.dot(np.array(heap[x-1][i][1]), np.array(heap[x-1][j][1])) / (np.linalg.norm(np.array(heap[x-1][i][1])) * np.linalg.norm(np.array(heap[x-1][j][1])))))
                            if theta > 30:
                                control_point = np.array(solution_space)
                                f_i = objective_functions[x-1](*heap[x-1][i][1])
                                heap[x-1][i] = ((1 - d_ij / 2) ** 2 * objective_functions[x-1](*heap[x-1][j][1]) + 2 * d_ij * (1 - d_ij) * control_point + d_ij ** 2 * f_i, heap[x-1][i][1])

            # Algorithm 4: Solution Ranking
            O_t = np.mean([heap[x-1][i][0] for i in range(len(heap[x-1]))])
            # print(O_t)
            if np.any(O_t <= heap[x-1][0][0]):  # For min-heap
                new_sol = O_t
                heapq.heappush(heap[x-1], (new_sol, solution_space))
                while new_sol <= heap[x-1][0][0]:
                    heapq.heappop(heap[x-1])
                    heapq.heappush(heap[x-1], (new_sol, solution_space))

    # After the loop ends, you can check if the termination was due to reaching the maximum iterations
    if t >= max_iterations:
        print("Maximum iterations reached...")
    else:
        print("Destination reached...")

    return heap

# USAGE
if __name__ == "__main__":
    num_objectives = 2  # Number of objective functions
    objective_functions = [objective_function_1, objective_function_2]  # List of objective functions
    relevance= [0.7, 0.3]  # Relevance of functions
    total_particles = 1000  # Total number of particles
    initial_solutions = np.random.rand(num_objectives, 3)  # Initial solutions in each solution space
    solution_space = np.random.rand(3)  # 3-dimensional cartesian solution space
    min_solutions = []

    optimized_solutions_heap = great_migration_herd_optimization(num_objectives, objective_functions, relevance, total_particles, initial_solutions, solution_space)

    # Extract optimized solutions from the heap
    optimized_solutions = []
    while optimized_solutions_heap:
        optimized_solutions.append(heapq.heappop(optimized_solutions_heap))

    # Print or use optimized_solutions as needed
    print("\n -- Optimized Solutions Track:")
    for i, solution in enumerate(optimized_solutions):
        print(f"Solution {i+1}: {solution}")
        min_solutions.append(solution[0])

    # Find Optimal solutions for the different objective functions
    min_solutions = np.mean(np.array(min_solutions), axis=1)

    # Show optimal solutions for each objective function
    for i in range(len(min_solutions)):
        print(f"\t Optimal Solution {i}: {min_solutions[i]}")
