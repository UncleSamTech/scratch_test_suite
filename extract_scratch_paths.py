import networkx as nx
import sqlite3
import json
import matplotlib.pyplot as plt
from scratch_revisions_extractor import get_connection2

class Scratch_Path:

    def __init__(self):
        self.node = None
        self.root = None
        self.local_conn = get_connection2()
        self.all_hashes = []
        self.all_connections = None
        self.all_nodes = None

    def get_connection(self):
        conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database2.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor
    
    def get_all_contents(self,hash):
        
        conn,curr = self.local_conn
        if conn != None:
         curr.execute("select content from contents where hash = ? ", (hash,))  
         val = curr.fetchall()[0][0]
        return val

    def get_all_hashes(self,file_path):
        with open(file_path,"r") as hash:
            all_hashes = hash.readlines()
            for each_hash in all_hashes:
                each_hash = each_hash.strip()
                self.all_hashes.append(each_hash)
        return self.all_hashes
    
    def add_scratch_edges_recursive(self,sc_gr,node):
        sc_gr = nx.DiGraph()
        if isinstance(node,list):
            for i in range(len(node) - 1):
                sc_gr.add_edge(node[i],node[i + 1])
                self.add_scratch_edges_recursive(sc_gr,node[i + 1])
    
    def create_graph(self,all_connections):
        scratch_graph = nx.DiGraph()
        if isinstance(all_connections,list) and len(all_connections) > 0:
            for node in all_connections:
                if isinstance(node,list):
                    scratch_graph.add_nodes_from(node)
                    self.add_scratch_edges_recursive(node)
                else:
                    scratch_graph.add_nodes_from(node)
        return scratch_graph

    def generate_simple_graph(self,file_path):
        hashes = self.get_all_hashes(file_path)
        if len(hashes) > 0:
            for each_hash in hashes:
                contents = json.loads(self.get_all_contents(each_hash))
                
                try:
                    self.all_connections = contents["stats"]["connections"]
                    self.all_nodes = contents["stats"]["all_nodes"]
                    print(self.all_connections)
                    if len(self.all_nodes) > 0:
                        pass
                        #for each_connection in self.all_connections:
                        #graph = self.create_graph(['CS50 - Problem Set 0 v2 (1)', 'event_whenflagclicked', 'control_forever', 'BodyBlock', 'control_wait'])
                        #print(graph)
                except:
                    self.all_connections = []
                    self.all_nodes = []
                
                       
        return self.all_connections
    
    def visualize_graph(self,graph):
        sc_gr_pos = nx.spring_layout(graph)
        nx.draw(graph,sc_gr_pos,with_labels=True,arrows=True)
        plt.show()
    
sc_path = Scratch_Path()
#print(sc_path.get_all_hashes("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
#print(sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
gr = sc_path.create_graph(['CS50 - Problem Set 0 v2 (1)', 'event_whenflagclicked', 'control_forever', 'BodyBlock', 'control_wait'])
print(gr)
#vis = sc_path.visualize_graph(gr)
#print(vis)