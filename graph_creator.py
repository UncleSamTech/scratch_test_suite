import os
import json
from scratch_parser import scratch_parser

class graph_creator:
    def __init__(self):
        pass


    def flatten_tree(self,scratch_tree):
        flattened_list = []
        i = 0
        adj_list = {}
        stack = [scratch_tree[1:]]
        while stack:
            current = stack.pop()
            for val in current:
                if isinstance(val,list):
                    stack.append(val)
                else:
                    if val not in adj_list:
                        adj_list[val] = None
                    else:
                        i += 1
                        adj_list[f'{val}#{i}'] = None
        return adj_list
    
    def list_to_dict(self,lst):
        if not isinstance(lst, list):
            return lst
        elif len(lst) == 2 and isinstance(lst[0], str):
           
            return {lst[0]: self.list_to_dict(lst[1])}
        else:
            
            return [self.list_to_dict(sublist) for sublist in lst]
    
    def implement_directed_graph(self,graph,current):
        if current not in graph:
            return None
        new_connections = []
        stack_store = [current]
        while len(stack_store) > 0:
            curr = stack_store.pop()
            new_connections.append(curr)
            
            if isinstance(curr,str) or isinstance(curr,int) or isinstance(curr,float) or isinstance(curr,bool):
                if isinstance(graph,dict) and curr in graph.keys() and isinstance(graph[curr],list):
                    for neigbour in graph[curr]:
                        stack_store.append({current:neigbour})
        new_connections.remove(current)
        return new_connections

    def get_leaf(self,nodes,tree_passed):
        leafs = []
        def _get_leaf_nodes(node,tree_passed):

            if node is not None:
                if len(tree_passed) == 1:
                    leafs.append(node)
                else:
                    for n in tree_passed:
                        _get_leaf_nodes(n,tree_passed)
        _get_leaf_nodes(nodes,tree_passed)
        return leafs

    def get_length(self,tree_val):
        return len(tree_val)
    
    def get_child(self,tree_val):  
        if tree_val is None or len(tree_val) == 0:
            return []
        results = [tree_val[0]]
        if len(tree_val[1:]) > 0:
            for child in tree_val[1:]:
                results.extend(self.get_child(child))
        return results
    
    def get_paths(self,node,visited,tree):
        if tree is None and node not in visited:
            return node
        visited.add(node if isinstance(node,str) or isinstance(node,int) or isinstance(node,bool) or isinstance(node,float) else "")
        for each_value in tree:
            self.get_paths(each_value, visited,tree)
    
    
sp = scratch_parser()
gp = graph_creator()
tr = sp.read_files("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/an_check_for.sb3")
#print(tr)
dr_graph = gp.list_to_dict(tr)
#print(dr_graph)
curr_graph = dr_graph
root = list(curr_graph.keys())
if len(root) > 0:
    imp_grap  = gp.implement_directed_graph(curr_graph,root[0])
    print(imp_grap)
print(tr)
#print(gp.get_child(tr))
visited = set()
#paths = gp.get_paths(tr[0],visited,tr[1:])
#print(paths)
#print(gp.get_leaf(tr[0],tr[1:]))
#print(gp.get_length(tr))
#print(gp.flatten_tree(tr))