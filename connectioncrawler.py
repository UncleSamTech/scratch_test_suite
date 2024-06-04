import random
import sys
import timeit

def all_paths_from(current,prefix,seen,edge_map):
    if prefix is None:
        prefix = []
    if seen is None:
        seen = {}
    paths = []
    for k in edge_map[current]:
        if k in seen:
            continue
        new_prefix = prefix + [k]
        paths.append( new_prefix )
        seen[k] = 1
        paths = paths + all_paths_from(k,new_prefix, seen, edge_map)
    seen[current] = 0
    return paths

def all_all_paths(nodes,edge_map):
    all_paths = []
    for i in nodes:
        all_paths = all_paths + (all_paths_from(i,[],{},edge_map))
    return all_paths

def make_edge_map(nodes,edges):
    edge_map = dict([(i,dict()) for i in nodes])
    for a,b in edges:
        edge_map[a][b] = 1
    return edge_map

def big_test(n=256):
        sys.setrecursionlimit(1500)
        def r(n=n):
            return random.randint(0,n-1)
        nodes = list(range(0,n))
        edges = [(r(),r()) for i in range(n*n)]
        edges = set(edges)
        edge_map = make_edge_map(nodes, edges)
        return all_all_paths(nodes, edge_map)

def small_test():
    nodes = [0,1,2,3,4]
    edges = [(0,1),(1,2),(2,3),(2,4)]
    edge_map = make_edge_map(nodes, edges)
    result =  all_all_paths(nodes, edge_map)
    expected = [[1], [1, 2], [1, 2, 3], [1, 2, 4], [2], [2, 3], [2, 4], [3], [4]]
    expected = [tuple(x) for x in expected]
    expected = set(expected)
    for x in result:
        assert tuple(x) in expected
    return result

print(small_test())
for s in [16,32,64,128,256,512]:
    code = f'big_test({s})'
    f = lambda: timeit.timeit(code,number=5,globals=globals())
    print(f'{code} - {f()}')
