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
    new = val["Total_Commits"].describe()
    resp = new.to_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/projects_commits_analysis_descriptions.csv")
    return resp

#merge_csv_files("/media/crouton/siwuchuk/newdir/vscode_repos_files/total_commits.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2.csv","projects")
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_unique.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Differences in Nodes Per Revision of a  Scratch(sb3) File","Number of Total Revisions (Log Scale)","Histogram of Difference in Nodes Per Revision of a Scratch(sb3) File","final_differences_nodes_distribution_per_file","Nodes",15)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_unique.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Differences in Edges Per Revision of a  Scratch(sb3) File","Number of Total Revisions (Log Scale)","Histogram of Difference in Edges Per Revision of a Scratch(sb3) File","final_differences_edges_distribution_per_file","Edges",20)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/all_revisions_explore.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Number of Nodes Per Scracth(sb3) File","Number of Scratch File (Log Scale)","Histogram of Number of Nodes Per Scratch(sb3) File","main_nodes_scratch_files_distribution_per_projects","Nodes",20)
#plot_histogram_per_distribution("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/all_revisions_explore.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/all_plots_results","Number of Edges Per Scracth(sb3) File","Number of Scratch File (Log Scale)","Histogram of Number of Edges Per Scratch(sb3) File","main_edges_scratch_files_distribution_per_projects2","Edges",25)
describe_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/projects_commits_analysis.csv")
#generate_cleaned_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_unique.csv","/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/differences_final_cleaned_without_zero_unique.csv")