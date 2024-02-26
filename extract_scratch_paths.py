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
    
   
    
    def create_graph(self,all_connections,all_nodes):
        scratch_graph = nx.DiGraph()
        scratch_graph.add_nodes_from(all_nodes)

        for i in range(len(all_connections) - 1):
            scratch_graph.add_edge(all_connections[i],all_connections[i + 1])
        return scratch_graph

    def generate_simple_graph(self,file_path,path_name):
        
        hashes = self.get_all_hashes(file_path)
            
        if len(hashes) > 0:
            for each_hash in hashes:
                each_hash = each_hash.strip() if isinstance(each_hash,str) else each_hash
                contents = self.get_all_contents(each_hash)
                if contents is None or len(contents) < 1:
                    continue
                else:
                    contents2 = json.loads(contents)
                    try:
                        self.all_connections = contents2["stats"]["connections"]
                        self.all_nodes = contents2["stats"]["all_nodes"]
                    except:
                        self.all_connections = []
                        self.all_nodes = []
                    if isinstance(self.all_connections,list) and len(self.all_connections) > 0:
                        with open(path_name + each_hash + ".txt","a") as fp:
                            for each_connection in self.all_connections:
                                for i in range(len(each_connection)):
                                    fp.write(each_connection[i] + " ")
                                fp.write("\n")
                    else: 
                        continue 
      
                
                       
        
    
    def visualize_graph(self,graph):
        sc_gr_pos = nx.spring_layout(graph)
        nx.draw(graph,sc_gr_pos,with_labels=True,arrows=True)
        plt.show()
    
sc_path = Scratch_Path()
#print(sc_path.get_all_hashes("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
#print(sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))

sc_path.generate_simple_graph("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/sc_hash_local.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/extracted_paths/")
#sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/sb3_parsed/extracted_paths/")

#v = sc_path.get_all_contents("cfbab365b6dd7f4138823df8ff2e89a108f43dbf8c9950ab27ac8cc981b9adac")
#vis = sc_path.visualize_graph(gr)
#print('contents',v)
#print(vis)