import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_analysis.db")
cursor = connection.cursor()

'''

Distribution of nodes and edges

'''

cursor.execute('''SELECT Nodes, Edges FROM Revisions;''')
nodes_edges = cursor.fetchall()

# Create a DataFrame

df = pd.DataFrame(nodes_edges, columns=['Nodes', 'Edges'])
nodes = df['Nodes'].values
edges = df['Edges'].values
print(df["Nodes"].describe())
print(df["Edges"].describe())


cursor.execute('''SELECT COUNT(nodes) FROM revisions WHERE nodes >= 1 and nodes <= 50;''')
num_nodes_1_50 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/nodes_btw_1_50.csv","a") as nd150:
    for each_val in num_nodes_1_50:
        nd150.write(f"{each_val}\n")
    
print("Number of nodes between 1 and 50: ", num_nodes_1_50)


cursor.execute('''SELECT COUNT(nodes) FROM revisions WHERE nodes >= 100 and nodes <= 500;''')
num_nodes_100_500 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/nodes_btw_100_500.csv","a") as nd100500:
    for each_val in num_nodes_100_500:
        nd100500.write(f"{each_val}\n")
    
print("Number of nodes between 100 and 500: ", num_nodes_100_500)

cursor.execute('''SELECT COUNT(edges) FROM revisions WHERE edges >= 1 and edges <= 50;''')
num_edges_1_50 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_btw_1_50.csv","a") as ed150:
    for each_edge in num_edges_1_50:
        ed150.write(f"{each_edge}\n")

print("Number of edges between 1 and 50: ", num_edges_1_50)

cursor.execute('''SELECT COUNT(edges) FROM revisions WHERE edges >= 100 and edges <= 500;''')
num_edges_100_500 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_btw_100_500.csv","a") as ed100500:
    for each_edge in num_edges_100_500:
        ed150.write(f"{each_edge}\n")

print("Number of edges between 100 and 500: ", num_edges_100_500)

cursor.execute('''SELECT COUNT(nodes) FROM revisions WHERE nodes = 0;''')
num_nodes_0 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/nodes_btw_0.csv","a") as nd0:
    for each_node in num_nodes_0:
        nd0.write(f"{each_node}\n")

print("Number of nodes = 0: ", num_nodes_0)

cursor.execute('''SELECT COUNT(edges) FROM revisions WHERE edges = 0;''')
num_edges_0 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_btw_0.csv","a") as ned0:
    for each_edge in num_edges_0:
        nd0.write(f"{each_edge}\n")

print("Number of edges = 0: ", num_edges_0)

cursor.execute('''SELECT COUNT(edges) FROM revisions WHERE edges >= 500;''')
num_edges_gt_500 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_btw_0.csv","a") as edgt500:
    for each_edge in num_edges_gt_500:
        edgt500.write(f"{each_edge}\n")

print("Number of edges >= 500: ", num_edges_gt_500)

