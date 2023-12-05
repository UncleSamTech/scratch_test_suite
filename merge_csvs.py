import pandas as pd

# create the hash_nodes_edges_final.csv
df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/hashcontents/project_file_commit_hash2.csv')
df_nodes_edges = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/nodes_edges_per_file2.csv')
merged_df = pd.merge(df, df_nodes_edges, on=['Commit_SHA', 'Project_Name', 'File']) 


# save the dataframe df to a csv file
merged_df.to_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/merged_csv/project_file_commitsha_hash_nodes_edges_final.csv', index=False)


# create the project_file_revision_commitsha_commitdate_hash_nodes_edges_final.csv (the csv for the revision table)
df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/merged_csv/project_file_commitsha_hash_nodes_edges_final.csv')
df2 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/project_file_revision_commitsha_commitdate_alter.csv')
merged_df2 = pd.merge(df2, df, on=['Commit_SHA', 'Project_Name', 'File'])


# save the dataframe df2 to a csv file
merged_df2.to_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/merged_csv/project_file_revision_commitsha_commitdate_hash_nodes_edges_final.csv', index=False)
print(len(merged_df2))