import matplotlib.pyplot as plt
import pandas as pd
import os



def get_nodes_edges_per_file_distribution(file_path:str,plot_result_path):
    df = pd.read_csv(file_path)
    nodes = df['Nodes'].values
    edges = df['Edges'].values

    plt.hist(nodes,color='lightblue', ec='black',bins=20)
    plt.yscale('log')
    plt.ticklabel_format(axis='x',style='plain')
    plt.xlabel('Number of Nodes Per File')
    plt.ylabel('Number of Files (Log Scale)')
    plt.title('Histogram of Number of Nodes Per Scratch(.sb3) File')
    plt.show()
    return plt.savefig(f'{plot_result_path}/nodes_edges_distribution.pdf')


get_nodes_edges_per_file_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/nodes_edges_per_file2.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results")