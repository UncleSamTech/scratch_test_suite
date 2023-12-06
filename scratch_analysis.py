import matplotlib.pyplot as plt
import pandas as pd
import os



def plot_histogram_per_distribution(file_path:str,plot_result_path,xlabel,ylabel,title,fig_title):
    df = pd.read_csv(file_path)
    nodes = df['Nodes'].values
    edges = df['Edges'].values

    plt.hist(nodes,color='lightblue', ec='black',bins=20)
    plt.yscale('log')
    plt.ticklabel_format(axis='x',style='plain')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return plt.savefig(f'{plot_result_path}/{fig_title}.pdf')


plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/nodes_edges_per_project2.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Number of Nodes Per Projects","Number of Projects (Log Scale)","Histogram of Number of Nodes Per Scratch Project","nodes_edges_distribution_per_projects")