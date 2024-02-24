import networkx as nx
import sqlite3
import json
import matplotlib.pyplot as plt


class Scratch_Path:

    def __init__(self):
        self.node = None
        self.root = None
        self.all_hashes = []
        self.all_connections = None
        self.all_paths = []
        self.final_paths = []
        self.all_nodes = None

    def get_connection(self):
        conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database2.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor
    
    def get_connection2(self):
        conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_database.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor

    
    def get_all_contents(self,hash):
        
        conn,curr = self.get_connection2()
        if conn != None:
         curr.execute("select distinct(content) from contents where hash = ? ", (hash,))  
         try:
            val = curr.fetchall()[0][0]
            #fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
            
         except Exception as e:
             print(e.with_traceback)
         
         return val

    def get_all_hashes(self,file_path):
        with open(file_path,"r") as hash:
            all_hashes = hash.readlines()
            if len(all_hashes) > 0:
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
    
    def create_graph(self,all_connections,all_nodes):
        scratch_graph = nx.DiGraph()
        scratch_graph.add_nodes_from(all_nodes)

        for i in range(len(all_connections) - 1):
            scratch_graph.add_edge(all_connections[i],all_connections[i + 1])
        '''
        if isinstance(all_connections,list) and len(all_connections) > 0:
            for node in all_connections:
                if isinstance(node,list):
                    scratch_graph.add_nodes_from(node)
                    self.add_scratch_edges_recursive(node)
                else:
                    scratch_graph.add_nodes_from(node)
        '''
        return scratch_graph

    def generate_simple_graph(self,file_path):
        try:
            hashes = self.get_all_hashes(file_path)
            
            if len(hashes) > 0:
                for each_hash in hashes:
                    each_hash = each_hash.strip() if isinstance(each_hash,str) else each_hash
                    contents = json.loads(self.get_all_contents(each_hash))
                    #check = list(set(contents["stats"]["connections"]))
                    #print('see',check)
                    self.all_connections = contents["stats"]["connections"]
                    self.all_nodes = contents["stats"]["all_nodes"]
                    #print(self.all_connections)
                    if len(self.all_nodes) > 0:
                        
                        for each_connection in self.all_connections:
                            graph = self.create_graph(each_connection,self.all_nodes)
                            root = each_connection[0]
                            leaf = each_connection[-1]
                            all_paths = nx.all_simple_paths(graph,root,leaf,5)
                            self.all_paths.extend(all_paths)

                           
        except:
                self.all_connections = []
                self.all_nodes = []
                
                       
        return self.all_paths
    
    def visualize_graph(self,graph):
        sc_gr_pos = nx.spring_layout(graph)
        nx.draw(graph,sc_gr_pos,with_labels=True,arrows=True)
        plt.show()
    
sc_path = Scratch_Path()
#print(sc_path.get_all_hashes("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
#print(sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
gr = sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt")
print(gr)
#v = sc_path.get_all_contents("cfbab365b6dd7f4138823df8ff2e89a108f43dbf8c9950ab27ac8cc981b9adac")
#vis = sc_path.visualize_graph(gr)
#print('contents',v)
#print(vis)