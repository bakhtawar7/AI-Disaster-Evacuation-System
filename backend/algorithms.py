import heapq
import math
import random
from collections import deque
from typing import List, Tuple, Dict, Optional


# ─── Graph / Grid Utilities ────────────────────────────────────────────────────

def euclidean(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def get_neighbors(pos, grid, allow_diagonal=False):
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if allow_diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    result = []
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            cost = 1.414 if (dr != 0 and dc != 0) else 1
            # hazard cells cost more
            cell = grid[nr][nc]
            if cell == 3:   # fire/flood
                cost *= 5
            elif cell == 4: # congested road
                cost *= 2
            result.append(((nr, nc), cost))
    return result

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


# ─── 1. A* ─────────────────────────────────────────────────────────────────────

def astar(grid, start, goal, heuristic='euclidean'):
    h = euclidean if heuristic == 'euclidean' else manhattan
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: h(start, goal)}
    visited = []

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.append(current)
        if current == goal:
            return reconstruct_path(came_from, current), visited, g[current]
        for neighbor, cost in get_neighbors(current, grid, allow_diagonal=True):
            tentative_g = g[current] + cost
            if tentative_g < g.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f[neighbor], neighbor))
    return [], visited, float('inf')


# ─── 2. IDA* ───────────────────────────────────────────────────────────────────

def idastar(grid, start, goal, heuristic='euclidean'):
    h = euclidean if heuristic == 'euclidean' else manhattan
    threshold = h(start, goal)
    path = [start]
    visited_all = []

    def search(path, g, threshold):
        current = path[-1]
        f = g + h(current, goal)
        visited_all.append(current)
        if f > threshold:
            return f, None
        if current == goal:
            return -1, list(path)
        minimum = float('inf')
        for neighbor, cost in get_neighbors(current, grid, allow_diagonal=True):
            if neighbor not in path:
                path.append(neighbor)
                t, result = search(path, g + cost, threshold)
                if t == -1:
                    return -1, result
                if t < minimum:
                    minimum = t
                path.pop()
        return minimum, None

    for _ in range(100):
        t, result = search(path, 0, threshold)
        if t == -1:
            total_cost = sum(
                euclidean(result[i], result[i+1]) for i in range(len(result)-1)
            )
            return result, visited_all, total_cost
        if t == float('inf'):
            return [], visited_all, float('inf')
        threshold = t
    return [], visited_all, float('inf')


# ─── 3. BFS ────────────────────────────────────────────────────────────────────

def bfs(grid, start, goal):
    queue = deque([(start, [start])])
    visited = {start}
    visited_order = []

    while queue:
        current, path = queue.popleft()
        visited_order.append(current)
        if current == goal:
            return path, visited_order, len(path) - 1
        for neighbor, _ in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return [], visited_order, float('inf')


# ─── 4. DFS ────────────────────────────────────────────────────────────────────

def dfs(grid, start, goal):
    stack = [(start, [start])]
    visited = set()
    visited_order = []

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)
        if current == goal:
            return path, visited_order, len(path) - 1
        for neighbor, _ in get_neighbors(current, grid):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return [], visited_order, float('inf')


# ─── 5. Best-First Search ──────────────────────────────────────────────────────

def best_first_search(grid, start, goal):
    h = euclidean
    open_set = [(h(start, goal), start)]
    came_from = {}
    visited = set()
    visited_order = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))
            return path, visited_order, cost
        for neighbor, _ in get_neighbors(current, grid, allow_diagonal=True):
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(open_set, (h(neighbor, goal), neighbor))
    return [], visited_order, float('inf')


# ─── 6. Minimax (Evacuee vs Spreading Disaster) ────────────────────────────────

def minimax_evacuation(grid, start, goal, hazard_sources, depth=4):
    """
    MAX player = evacuee (wants to reach goal)
    MIN player = disaster (spreads to block paths)
    Returns best next move for evacuee.
    """
    rows, cols = len(grid), len(grid[0])
    visited_order = []

    def spread_hazard(hazards, g):
        new_g = [row[:] for row in g]
        new_hazards = set(hazards)
        for (hr, hc) in hazards:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = hr+dr, hc+dc
                if 0 <= nr < rows and 0 <= nc < cols and new_g[nr][nc] == 0:
                    new_g[nr][nc] = 3
                    new_hazards.add((nr, nc))
        return new_g, new_hazards

    def minimax(pos, hazards, g, depth, is_max):
        visited_order.append(pos)
        if pos == goal:
            return 1000, pos
        if g[pos[0]][pos[1]] == 3:
            return -1000, pos
        if depth == 0:
            score = -euclidean(pos, goal)
            return score, pos

        if is_max:
            best = (-float('inf'), pos)
            for neighbor, _ in get_neighbors(pos, g):
                score, _ = minimax(neighbor, hazards, g, depth-1, False)
                if score > best[0]:
                    best = (score, neighbor)
            return best
        else:
            new_g, new_hazards = spread_hazard(hazards, g)
            score, move = minimax(pos, new_hazards, new_g, depth-1, True)
            return score, move

    _, best_move = minimax(start, set(hazard_sources), grid, depth, True)

    # Build a path greedily using best-first after minimax decision
    path, vis, cost = astar(grid, best_move, goal)
    full_path = [start] + path if best_move != start else path
    return full_path, visited_order + vis, cost


# ─── 7. Genetic Algorithm (Shelter Placement) ──────────────────────────────────

def genetic_algorithm_shelters(
    grid, population_positions, num_shelters=3,
    generations=50, pop_size=30, mutation_rate=0.15
):
    rows, cols = len(grid), len(grid[0])
    free_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]

    def fitness(chromosome):
        """Lower average distance from all people to nearest shelter = better"""
        total = 0
        for person in population_positions:
            min_dist = min(euclidean(person, shelter) for shelter in chromosome)
            total += min_dist
        return 1 / (total + 1e-9)

    def random_individual():
        return random.sample(free_cells, min(num_shelters, len(free_cells)))

    def crossover(p1, p2):
        cut = random.randint(1, num_shelters - 1)
        child = p1[:cut]
        for gene in p2:
            if gene not in child:
                child.append(gene)
            if len(child) == num_shelters:
                break
        while len(child) < num_shelters:
            candidate = random.choice(free_cells)
            if candidate not in child:
                child.append(candidate)
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, num_shelters - 1)
            new_gene = random.choice(free_cells)
            individual[idx] = new_gene
        return individual

    # Init population
    population = [random_individual() for _ in range(pop_size)]
    best_history = []

    for gen in range(generations):
        scored = sorted(population, key=fitness, reverse=True)
        best_history.append({
            "generation": gen,
            "fitness": round(fitness(scored[0]), 6),
            "shelters": scored[0]
        })
        # Elitism + crossover
        elite = scored[:pop_size//4]
        new_pop = elite[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elite, 2)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop

    best = max(population, key=fitness)
    return best, best_history
