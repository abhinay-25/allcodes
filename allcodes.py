#A * algorithm
#A* algorithm 
import heapq

def astar(start, goal, graph, h):
    open_list = [(h[start], 0, start, [start])]
    closed = set()
    while open_list:
        f, g, node, path = heapq.heappop(open_list)
        if node == goal:
            return path, g
        if node in closed: 
            continue
        closed.add(node)
        for neigh, cost in graph[node]:
            if neigh not in closed:
                heapq.heappush(open_list, (g+cost+h[neigh], g+cost, neigh, path+[neigh]))
    return None

# ----- Dynamic Input -----
n = int(input("Enter number of nodes: "))
graph = {}
for _ in range(n):
    node = input("Enter node: ")
    edges = eval(input(f"Enter edges for {node} as (neighbor,cost) list: "))
    graph[node] = edges

h = eval(input("Enter heuristic dict (e.g., {'A':3,'B':2,'C':0}): "))
start = input("Enter start node: ")
goal = input("Enter goal node: ")

path, cost = astar(start, goal, graph, h)
print("Path:", path)
print("Cost:", cost)


#manhanttan distance heuristic
import heapq

def manhattan(state, goal):
    dist=0
    for n in range(1,9):
        i,j=state.index(n),goal.index(n)
        dist+=abs(i//3-j//3)+abs(i%3-j%3)
    return dist

def astar(start, goal):
    pq=[(manhattan(start,goal),0,start,[start])]
    seen={start:0}
    while pq:
        f,g,state,path=heapq.heappop(pq)
        if state==goal:
            for i,s in enumerate(path):
                print(f"Move {i}:\n{s[0]} {s[1]} {s[2]}\n{s[3]} {s[4]} {s[5]}\n{s[6]} {s[7]} {s[8]}\n")
            return g
        i=state.index(0)
        for d in [-3,3,-1,1]:
            j=i+d
            if 0<=j<9 and not(i%3==2 and d==1) and not(i%3==0 and d==-1):
                new=list(state)
                new[i],new[j]=new[j],new[i]
                new=tuple(new)
                if new not in seen or g+1<seen[new]:
                    seen[new]=g+1
                    heapq.heappush(pq,(g+1+manhattan(new,goal),g+1,new,path+[new]))

start = tuple(map(int,input("enter start").split()))
goal  = tuple(map(int,input("enter goal").split()))
print("Moves:", astar(start,goal))

#misplaced
import heapq

def misplaced(state, goal): 
    return sum(s != g and s != 0 for s, g in zip(state, goal))

def astar(start, goal):
    pq = [(misplaced(start, goal), 0, start, [start])]
    seen = {start:0}
    while pq:
        f, g, state, path = heapq.heappop(pq)
        if state == goal:
            for i, s in enumerate(path):
                print(f"Move {i}:")
                print(f"{s[0]} {s[1]} {s[2]}\n{s[3]} {s[4]} {s[5]}\n{s[6]} {s[7]} {s[8]}\n")
            return g
        i = state.index(0)
        for d in [-3,3,-1,1]:
            j=i+d
            if 0<=j<9 and not(i%3==2 and d==1) and not(i%3==0 and d==-1):
                new=list(state)
                new[i],new[j]=new[j],new[i]
                new=tuple(new)
                if new not in seen or g+1<seen[new]:
                    seen[new]=g+1
                    heapq.heappush(pq,(g+1+misplaced(new,goal),g+1,new,path+[new]))

start = tuple(map(int,input("enter start").split()))
goal  = tuple(map(int,input("enter goal").split()))
print("Moves:", astar(start,goal))

#map-coloring
def csp_coloring(graph, colors, assignment={}, i=0):
    if i == len(graph):
        return assignment
    
    node = list(graph)[i]
    for c in colors:
        if all(assignment.get(n) != c for n in graph[node]):
            assignment[node] = c
            res = csp_coloring(graph, colors, assignment, i+1)
            if res: return res
            assignment.pop(node)
    return None



graph = {}
for _ in range(int(input("Enter number of edges: "))):
    u, v = input("Edge (u v): ").split()
    graph.setdefault(u, []).append(v)
    graph.setdefault(v, []).append(u)

colors = input("Enter colors (space separated): ").split()

print("\nSolution:", csp_coloring(graph, colors))

#cryptarithmetic
import itertools

def solve(expr):
    left, result = expr.split("=")
    words = left.split("+")
    letters = list(set("".join(words) + result))
    
    for perm in itertools.permutations("0123456789", len(letters)):
        mapping = dict(zip(letters, perm))
        
        if any(mapping[word[0]] == '0' for word in words + [result]):
            continue
        
        values = [int("".join(mapping[ch] for ch in word)) for word in words]
        res_value = int("".join(mapping[ch] for ch in result))
        
        if sum(values) == res_value:
            return {letter: int(digit) for letter, digit in mapping.items()}
    
    return None

expr = input("Enter cryptarithm: ").replace(" ", "")
solution = solve(expr)
print("Solution:", solution)

#minmax
def minimax(depth, node_index, scores, h, level_order):
    if depth == h:       
        return scores[node_index]
    
    if level_order[depth] == "max":  
        return max(minimax(depth+1, node_index*2, scores, h, level_order),
                   minimax(depth+1, node_index*2+1, scores, h, level_order))
    else: 
        return min(minimax(depth+1, node_index*2, scores, h, level_order),
                   minimax(depth+1, node_index*2+1, scores, h, level_order))



scores = list(map(int, input("Enter leaf node values: ").split()))
import math
h = int(math.log2(len(scores)))
level_order = input(f"Enter level order (space separated for {h} levels from bottom): ").split()
best = minimax(0, 0, scores, h, level_order)
print("Optimal value:", best)

#alpha beta
def alpha_beta(depth, node_index, scores, h, level_order, alpha, beta):
    if depth == h:
        return scores[node_index]

    if level_order[depth] == "max":  
        value = float("-inf")
        for child in range(2):
            val = alpha_beta(depth+1, node_index*2+child, scores, h, level_order, alpha, beta)
            value = max(value, val)
            alpha = max(alpha, value)
            if alpha >= beta:  
                break
    else:                       
        value = float("inf")
        for child in range(2):
            val = alpha_beta(depth+1, node_index*2+child, scores, h, level_order, alpha, beta)
            value = min(value, val)
            beta = min(beta, value)
            if alpha >= beta:   
                break
    return value


import math
scores = list(map(int, input("Leaf values: ").split()))
h = int(math.log2(len(scores)))
level_order = input(f"Enter order for {h} levels (e.g. max min max): ").split()

print("Optimal value:", alpha_beta(0, 0, scores, h, level_order, float("-inf"), float("inf")))
