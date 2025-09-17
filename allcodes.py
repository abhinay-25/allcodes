# A* Search Algorithm Implementation with Dynamic Input
from queue import PriorityQueue

def a_star_search(graph, start, goal, heuristic):
    open_list = PriorityQueue()
    open_list.put((0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while not open_list.empty():
        f, current = open_list.get()

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor, cost in graph[current]:
            new_g = g_cost[current] + cost

            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f_value = new_g + heuristic[neighbor]
                open_list.put((f_value, neighbor))
                parent[neighbor] = current

    return None


# -------- Dynamic Input --------
graph = {}
n = int(input("Enter number of nodes: "))

for i in range(n):
    node = input(f"Enter node {i+1} name: ")
    edges = int(input(f"Enter number of neighbors of {node}: "))
    graph[node] = []
    for j in range(edges):
        neigh = input(f"  Neighbor {j+1} of {node}: ")
        cost = int(input(f"  Cost from {node} to {neigh}: "))
        graph[node].append((neigh, cost))

heuristic = {}
print("\nEnter heuristic values:")
for node in graph.keys():
    heuristic[node] = int(input(f"h({node}) = "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path = a_star_search(graph, start, goal, heuristic)
print("\nShortest Path:", path if path else "No Path Found")


#end

# uniform cost search
from queue import PriorityQueue

def uniform_cost_search(graph, start, goal):
    open_list = PriorityQueue()
    open_list.put((0, start))   # (cost, node)

    g_cost = {start: 0}
    parent = {start: None}

    while not open_list.empty():
        cost, current = open_list.get()

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1], g_cost[goal]

        for neighbor, edge_cost in graph[current]:
            new_cost = g_cost[current] + edge_cost

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                open_list.put((new_cost, neighbor))
                parent[neighbor] = current

    return None, float('inf')   # no path found


# -------- Dynamic Input --------
graph = {}
n = int(input("Enter number of nodes: "))

for i in range(n):
    node = input(f"Enter node {i+1} name: ")
    edges = int(input(f"Enter number of neighbors of {node}: "))
    graph[node] = []
    for j in range(edges):
        neigh = input(f"  Neighbor {j+1} of {node}: ")
        cost = int(input(f"  Cost from {node} to {neigh}: "))
        graph[node].append((neigh, cost))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

path, total_cost = uniform_cost_search(graph, start, goal)
if path:
    print("\nShortest Path:", path)
    print("Total Cost:", total_cost)
else:
    print("\nNo Path Found")

#end 
# 8-puzzle heuristic: Misplaced Tiles
def misplaced_tiles(state, goal):
    count = 0
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count


# ---------------- MAIN PROGRAM ----------------
print("Enter the initial 3x3 puzzle state (use 0 for blank):")
initial = []
for i in range(3):
    row = list(map(int, input().split()))
    initial.append(row)

print("\nEnter the goal 3x3 puzzle state (use 0 for blank):")
goal = []
for i in range(3):
    row = list(map(int, input().split()))
    goal.append(row)

print("\nInitial State:", initial)
print("Goal State:", goal)

# Heuristic calculation
misplaced = misplaced_tiles(initial, goal)
print("\nMisplaced Tiles Heuristic:", misplaced)

#end

# 8-puzzle heuristic: Manhattan Distance
def manhattan_distance(state, goal):
    distance = 0
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] != 0:  # skip blank tile
                # Find correct position of this tile in the goal state
                for x in range(len(goal)):
                    for y in range(len(goal[0])):
                        if goal[x][y] == state[i][j]:
                            distance += abs(i - x) + abs(j - y)
    return distance


# ---------------- MAIN PROGRAM ----------------
print("Enter the initial 3x3 puzzle state (use 0 for blank):")
initial = []
for i in range(3):
    row = list(map(int, input().split()))
    initial.append(row)

print("\nEnter the goal 3x3 puzzle state (use 0 for blank):")
goal = []
for i in range(3):
    row = list(map(int, input().split()))
    goal.append(row)

print("\nInitial State:", initial)
print("Goal State:", goal)

# Heuristic calculation
manhattan = manhattan_distance(initial, goal)
print("\nManhattan Distance Heuristic:", manhattan)

#end
# Cryptoarithmetic Puzzle Solver
from itertools import permutations

def solve_crypto(words, result):
    # Collect all unique letters
    letters = set("".join(words) + result)
    letters = list(letters)

    if len(letters) > 10:
        print("Too many unique letters! Cannot map to digits.")
        return

    for perm in permutations("0123456789", len(letters)):
        mapping = dict(zip(letters, perm))

        # No leading zeros allowed in any word
        if any(mapping[w[0]] == '0' for w in words + [result]):
            continue

        # Convert words to numbers
        word_values = [int("".join(mapping[ch] for ch in w)) for w in words]
        result_value = int("".join(mapping[ch] for ch in result))

        # Check if sum of words equals result
        if sum(word_values) == result_value:
            print("\n✅ Solution Found:")
            for w, val in zip(words, word_values):
                print(f"{w} = {val}")
            print(f"{result} = {result_value}")
            return

    print("❌ No solution found.")

# ---------------- MAIN ----------------
equation = input("Enter cryptoarithmetic equation (example: SEND+MORE=MONEY): ").replace(" ", "")
left, right = equation.split("=")
words = left.split("+")
result = right

solve_crypto(words, result)

#end
# Map Coloring Problem Solver
def is_safe(node, color, assignment, graph):
    for neighbor in graph[node]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

def map_coloring(graph, colors, assignment, nodes, index=0):
    if index == len(nodes):
        return assignment  # all nodes colored successfully

    node = nodes[index]
    for color in colors:
        if is_safe(node, color, assignment, graph):
            assignment[node] = color
            result = map_coloring(graph, colors, assignment, nodes, index + 1)
            if result:
                return result
            assignment.pop(node)  # backtrack
    return None

# ---------------- MAIN ----------------
graph = {}
n = int(input("Enter number of regions: "))

# input graph dynamically
for i in range(n):
    node = input(f"Enter region {i+1} name: ")
    neighbors = input(f"Enter neighbors of {node} (comma separated, leave blank if none): ")
    graph[node] = neighbors.split(",") if neighbors else []

colors = input("Enter available colors (comma separated): ").split(",")

nodes = list(graph.keys())
solution = map_coloring(graph, colors, {}, nodes)

print("\n✅ Coloring Solution:" if solution else "\n❌ No solution found")
if solution:
    for region, color in solution.items():
        print(f"{region} → {color}")

#end
# Minimax Algorithm Implementation
def minimax(depth, node_index, is_max, scores, height):
    # Base case: leaf node
    if depth == height:
        return scores[node_index]

    if is_max:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, height),
            minimax(depth + 1, node_index * 2 + 1, False, scores, height)
        )
    else:
        return min(
            minimax(depth + 1, node_index * 2, True, scores, height),
            minimax(depth + 1, node_index * 2 + 1, True, scores, height)
        )

# ---------------- MAIN ----------------
import math

# Dynamic input
scores = list(map(int, input("Enter leaf node values (space separated): ").split()))
n = len(scores)

# Height of tree = log2(n)
height = math.log2(n)

if height.is_integer():
    result = minimax(0, 0, True, scores, int(height))
    print("\nOptimal value (using Minimax):", result)
else:
    print("❌ Number of leaf nodes must be a power of 2.")

#end

#alpha beta pruning
def alphabeta(depth, node_index, is_max, scores, alpha, beta, height):
    # Base case: leaf node
    if depth == height:
        return scores[node_index]

    if is_max:
        best = float('-inf')
        # left child
        val = alphabeta(depth + 1, node_index * 2, False, scores, alpha, beta, height)
        best = max(best, val)
        alpha = max(alpha, best)
        if beta <= alpha:   # prune
            return best

        # right child
        val = alphabeta(depth + 1, node_index * 2 + 1, False, scores, alpha, beta, height)
        best = max(best, val)
        alpha = max(alpha, best)
        return best
    else:
        best = float('inf')
        # left child
        val = alphabeta(depth + 1, node_index * 2, True, scores, alpha, beta, height)
        best = min(best, val)
        beta = min(beta, best)
        if beta <= alpha:   # prune
            return best

        # right child
        val = alphabeta(depth + 1, node_index * 2 + 1, True, scores, alpha, beta, height)
        best = min(best, val)
        beta = min(beta, best)
        return best


# ---------------- MAIN ----------------
import math

# Dynamic input
scores = list(map(int, input("Enter leaf node values (space separated): ").split()))
n = len(scores)

# Height of tree = log2(n)
height = math.log2(n)

if height.is_integer():
    result = alphabeta(0, 0, True, scores, float('-inf'), float('inf'), int(height))
    print("\nOptimal value (using Alpha-Beta Pruning):", result)
else:
    print("❌ Number of leaf nodes must be a power of 2.")

#end
