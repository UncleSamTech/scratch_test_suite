import networkx as nx
import sqlite3
import json

connection = sqlite3.connect("/media/crouton/aislam4/database_updated_11_dec/revisions4/revision_hashes.db")
cursor = connection.cursor()


def get_content_from_db(hash):
    cursor.execute("SELECT Content FROM Contents WHERE hash = ?", (hash,))
    content = cursor.fetchall()[0][0]
    return content

with open("Hashes.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        content = get_content_from_db(line)
        data = json.loads(content)

        try:
            connections = data["connections"]
            all_objects = data["all_objects"]
        except:
            connections = []
            all_objects = []

        if len(connections) > 0:

            object_dict = {}
            for obj in all_objects:
                if obj["box"]["object_type"] in ["list"]:
                    obj_text = obj["box"]["text"].split(" ")[:2]
                    obj_text_str = "_".join(obj_text)
                    object_dict[obj["box"]["id"]] = obj_text_str
                else:
                    object_dict[obj["box"]["id"]] = obj["box"]["object_type"]

            
            sources = [connection["patchline"]["source"][0] for connection in connections]
            destinations = [connection["patchline"]["destination"][0] for connection in connections]

            nodes = set(sources + destinations)
            
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from([(connection["patchline"]["source"][0], connection["patchline"]["destination"][0]) for connection in connections])

            roots = list(set(sources) - set(destinations))
            leaves = list(set(destinations) - set(sources))

            all_paths = []
            
            for root in roots:
                for leaf in leaves:
                    paths = nx.all_simple_paths(G, root, leaf, 5)
                    all_paths.extend(paths)
            
            # for all the elements in the all_paths, replace the entry with corresponding dictionary entry
            for path in all_paths:
                for i in range(len(path)):
                    path[i] = object_dict[path[i]]
            
            with open("/media/crouton/aislam4/paths/paths_5/" + line + ".txt", "a") as f:
                for path in all_paths:
                    for i in range(len(path)):
                        if i == len(path) - 1:
                            f.write(path[i])
                        else:
                            f.write(path[i] + " ")
                    f.write("\n")

        else:
            with open("/media/crouton/aislam4/paths/paths_5/" + line + ".txt", "w") as f:
                f.write("No connections")
            
            



cursor.close()
connection.close()
            
