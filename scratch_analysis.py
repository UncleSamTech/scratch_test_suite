import matplotlib.pyplot as plt
import pandas as pd
import os



def plot_histogram_per_distribution(file_path:str,plot_result_path,xlabel,ylabel,title,fig_title,column_title,bins_count):
    df = pd.read_csv(file_path)
    
    val = df[column_title].values
    
    
    plt.hist(val,color='lightblue', ec='black',bins=bins_count)
    plt.yscale('log')
    plt.ticklabel_format(axis='x',style='plain')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return plt.savefig(f'{plot_result_path}/{fig_title}.pdf')

def generate_cleaned_csv(file_path,cleaned_data_path):
    df = pd.read_csv(file_path)
    df = df[df.Nodes != 0]
    df = df[df.Edges != 0]
    return df.to_csv(cleaned_data_path)


def merge_csv_files(csv_file1,csv_file2,new_file_name):
    csv1 = pd.read_csv(csv_file1)
    csv2 = pd.read_csv(csv_file2)
    merged_data = pd.merge(csv1,csv2,on='Project_Name')
    return merged_data.to_csv(f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/{new_file_name}.csv',index=False)


def describe_data(csv_file):
    val = pd.read_csv(csv_file)
    #new = val[["Nodes","Edges"]].describe()
    new = val["Revision_Count"].describe()
    resp = new.to_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results/nodes_edges_differences_description.csv")

#merge_csv_files("/media/crouton/siwuchuk/newdir/vscode_repos_files/total_commits.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2.csv","projects")
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results/nodes_edges_differences.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results","Difference in node count of a commit and its content parents","Number of Total Revisions (Log Scale)","Histogram of Difference in node count","nodes_counts_response","Nodes",15)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results/nodes_edges_differences.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results","Difference in edge count of a commit and its content parents","Number of Total Revisions (Log Scale)","Histogram of Difference in edge count","edges_counts_response","Edges",20)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/all_nodes_edges_data.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Number of Nodes Per Scracth(sb3) File","Number of Scratch File (Log Scale)","Histogram of Number of Nodes Per Scratch(sb3) File","all_main_nodes_scratch_files_distribution_per_projects2","Nodes",20)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/authors_dataset_distributions.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results","Number of Authors Per Scratch Project (Without Revisions)","Number of Scratch Projects (Log Scale)","Histogram of Number of Authors Per Project (sb3)","all_authors_per_project","author_count",20)
describe_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/revisions_data_summary.csv")
#generate_cleaned_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_unique.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_without_zero_unique.csv")