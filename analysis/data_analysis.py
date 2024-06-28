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
#print(df["Nodes"].describe())
#print(df["Edges"].describe())

"""
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
        ed100500.write(f"{each_edge}\n")

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
        ned0.write(f"{each_edge}\n")

print("Number of edges = 0: ", num_edges_0)

cursor.execute('''SELECT COUNT(edges) FROM revisions WHERE edges >= 500;''')
num_edges_gt_500 = cursor.fetchall()

with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_gt_0.csv","a") as edgt500:
    for each_edge in num_edges_gt_500:
        edgt500.write(f"{each_edge}\n")

print("Number of edges >= 500: ", num_edges_gt_500)
"""

"""
plt.hist(nodes, color='lightblue', ec='black', bins=20)
plt.yscale('log')
plt.ticklabel_format(axis='x', style='plain')
plt.xlabel('Number of Nodes Per Revision of a Scratch3 File')
plt.ylabel('Number of Total Revisions of Scratch3 Files (Log Scale)')
plt.title('Histogram of Number of Nodes Per Revision of a Scratch3 File')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/nodes_per_file.pdf')
plt.close()

plt.hist(edges, color='lightblue', ec='black', bins=20)
plt.yscale('log')
plt.ticklabel_format(axis='x', style='plain')
plt.xlabel('Number of Connections Per Revision of a Scratch3 File')
plt.ylabel('Number of Total Revisions of Scratch3 Files (Log Scale)')
plt.title('Histogram of Number of Connections Per Revision of a Scratch3 File')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/edges_per_file.pdf')
plt.close()

"""

'''

Distribution of Revisions Per Scratch3 file

'''

"""

cursor.execute('''SELECT Project_Name, File, COUNT(Revision) AS Revision_Count FROM Revisions WHERE File IS NOT NULL AND FILE <> '' GROUP BY Project_Name, File;''')
revisions = cursor.fetchall()

df = pd.DataFrame(revisions, columns=['Project_Name', 'File', 'Revisions'])
data = df["Revisions"].values
print("Revisions per Scratch3 file ",df["Revisions"].describe())

plt.hist(data, color='lightblue', ec='black', bins=15)
plt.yscale('log')
plt.xlabel('Number of Revisions Per Scratch3 File')
plt.ylabel('Number of Scratch3 Files (Log Scale)')
plt.title('Histogram of Number of Revisions Per Scratch3 File')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/revisions_per_file.pdf')
plt.close()

'''

Distribution of Scratch3 files per project

'''


cursor.execute('''SELECT Project_Name, COUNT(DISTINCT File) AS File_Count FROM Revisions GROUP BY Project_Name;''')
files = cursor.fetchall()

df = pd.DataFrame(files, columns=['Project_Name', 'Files'])
data = df["Files"].values
print("Scratch3 Files Count Per Scratch3 Project", df["Files"].describe())
print("Scratch3 Files with length less than 20 ",len(df[df["Files"] <= 20]))

cursor.execute('''SELECT Project_Name, COUNT(DISTINCT File) AS File_Count FROM Revisions GROUP BY Project_Name ORDER BY File_Count DESC LIMIT 3;''')
highest_files = cursor.fetchall()
print("Scratch3 Projects with highest files: ", highest_files)


cursor.execute('''SELECT
    p.Project_Name,
    p.Total_Commits
FROM
    Projects p
JOIN (
    SELECT
        Project_Name,
        COUNT(DISTINCT File) AS File_Count
    FROM
        Revisions
    GROUP BY
        Project_Name
    ORDER BY
        File_Count DESC
    LIMIT 3
) r ON p.Project_Name = r.Project_Name;''')
highest_files_commits = cursor.fetchall()
print("Scratch3 Projects with highest files and their number of commits: ", highest_files_commits)

plt.hist(data, color='lightblue', ec='black', bins=20)
plt.yscale('log')
plt.ticklabel_format(axis='x', style='plain')
plt.xlabel('Number of Scratch3 Files Per Project (Without Revisions)')
plt.ylabel('Number of Projects (Log Scale)')
plt.title('Histogram of Number of Scratch3 Files Per Project (Without Revisions)')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/scratch3_files_per_project.pdf')
plt.close()
"""

'''

Commits per project distribution

'''


cursor.execute('''SELECT Total_Commits FROM Projects;''')
total_commits = cursor.fetchall()
df = pd.DataFrame(total_commits, columns=['Commits'])
df['Commits'] = df['Commits'].astype(int)
commits = df['Commits'].values
print("Commits per projects", df["Commits"].describe())

cursor.execute('''SELECT COUNT(total_commits) FROM projects WHERE total_commits <= 20;''')
commits_20 = cursor.fetchall()
print("Projects with <=20 commits: ", commits_20)

plt.hist(commits, color='lightgreen', ec='black', bins=20)
plt.yscale('log')
plt.ticklabel_format(axis='x', style='plain')
plt.xlabel('Number of Commits Per Project')
plt.ylabel('Number of Projects (Log Scale)')
plt.title('Histogram of Number of Commits Per Project')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/commits_per_project.pdf')
plt.close()


'''

Author who contributed to the Scratch3 files distribution per project

'''


cursor.execute('''SELECT Revisions.Project_Name, COUNT(DISTINCT Authors.Author_Name)
FROM Revisions
JOIN Authors ON Revisions.Commit_SHA = Authors.Commit_SHA
WHERE Revisions.file IS NOT NULL AND Revisions.file <> ''
GROUP BY Revisions.Project_Name;''')
author_count = cursor.fetchall()

df = pd.DataFrame(author_count, columns=['Project_Name', 'Authors'])
authors = df['Authors'].values
print("Authors contributution per Scratch3 file",df["Authors"].describe())


plt.hist(authors, color='lightblue', ec='black', bins=20)
plt.yscale('log')
plt.xlabel('Number of Authors Per Project For the Scratch3 Files')
plt.ylabel('Number of Projects (Log Scale)')
plt.title('Histogram of Number of Authors Per Project For the Scratch3 Files')
plt.savefig('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/analysis/Authors_per_project.pdf')
plt.close()


cursor.execute('''SELECT Revisions.Project_Name, COUNT(DISTINCT Authors.Author_Name) AS GroupSize
FROM Revisions
JOIN Authors ON Revisions.Commit_SHA = Authors.Commit_SHA
WHERE Revisions.file IS NOT NULL AND Revisions.file <> ''
GROUP BY Revisions.Project_Name
ORDER BY GroupSize DESC
LIMIT 5;''')
highest_author = cursor.fetchall()
print("Scratch3 Project with highest author count : ", highest_author)

cursor.execute('''SELECT Revisions.Project_Name, COUNT(DISTINCT Authors.Author_Name) AS AuthorCount
FROM Revisions
JOIN Authors ON Revisions.Commit_SHA = Authors.Commit_SHA
WHERE Revisions.file IS NOT NULL AND Revisions.file <> ''
GROUP BY Revisions.Project_Name
HAVING AuthorCount = 1;''')
author_1 = cursor.fetchall()
print("Scratch3 Projects with one authors: ", len(author_1))

cursor.close()
connection.commit()
connection.close()