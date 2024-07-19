import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import subprocess


conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_analysis.db')
#conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_test2.db",isolation_level=None)
list_of_implementation_keywords = [
    "implement", "implementation", "feature", "new feature", "add feature", 
    "add functionality", "extend feature", "introduce feature",
    "feature addition", "feature enhancement", "feature extension", "feature implementation",
    "feature update", "feature change", "feature improvement", "feature modification",
    "feature development", "feature revision", "feature iteration",
    "feature iteration", "feature evolution", "feature innovation",
    "feature enhancement","add", "add file","add project", "Add Scratch3","Add Scratch3","Add"
]

list_of_vcs_keywords=[
    "version", "release", "tag", "snapshot", "update", "changelog", "bump", "merge", "sync", "rebase", "squash",
    "conflict", "revert", "rollback", "history", "commit", "push", "pull", "fetch", "diff", "log", "annotate", 
    "amend", "checkout", "reset", "stage", "unstage", "status", "clean", "clone", "fork", "archive", "export", 
    "import", "save", "recover", "restore", "backup", "stash", "pop", "apply", "init", "set", "integrate",
    "branch", "feature", "hotfix", "develop", "master", "main", "stable", "integration", "checkout", "switch",
    "track", "untrack", "staging", "prod", "env", "pr", "merge request", "cherry-pick", "rename",
    "delete", "create", "split", "combine", "pull request", "deploy", "prepare", "migrate", "transition", "promote",
    "demote", "protect", "review", "approve", "reject", "feature toggle",
    "merge branch", "branch update", "branch sync", "version bump", "release branch", "merge conflict",
    "rebase branch", "branch checkout", "branch delete", "branch create", "branch rename", "branch switch",
    "branch track", "branch untrack", "branch fork", "branch clone", "branch clean", "branch save", "branch restore",
    "branch status", "branch protect", "branch review"
]

list_of_maintenance_keywords=[
    "maintenance", "cleanup", "refactor", "tidy", "housekeeping", "reorganize", "restructure", "format", "lint",
    "comment", "docs", "update dependencies", "dependency update", "deps", "upgrade", "downgrade",
    "remove", "delete", "deprecate", "obsolete", "cleanup unused", "unused", "rebuild", "configure", "settings",
    "optimize", "improve", "enhance", "speed up", "performance", "perf", "efficient", "reduce", "minimize",
    "maximize", "lightweight", "fast", "speed", "compress", "cache", "tune", "boost", "fine-tune", "profile",
    "debug", "fix", "bug", "issue", "error", "fault", "defect", "glitch", "correct", "resolve", "patch", "handle",
    "address", "troubleshoot", "repair", "solve", "diagnose", "root cause", "trace", "monitor", "alert",
    "exception", "catch", "handle exception", "fail", "crash", "recover",
    "bug fix", "error fix", "fix issue", "fix bug", "performance improvement", "optimize performance", "code cleanup",
    "refactor code", "cleanup code", "debug issue", "debug error", "fix crash", "resolve issue", "enhance performance",
    "optimize code", "error handling", "exception handling"
]

list_of_license_keywords=["legal", "license", "licensing", "copyright", "trademark", "patent", "compliance", "terms", "conditions",
    "policy", "agreement", "contract", "disclaimer", "notice", "attribution", "credits", "authors", "contributors",
    "ownership", "proprietary", "confidential", "privacy", "GDPR", "CCPA", "MIT", "Apache", "GPL", "LGPL", "BSD",
    "EULA", "FOSS", "OSS", "open source", "source code", "usage", "redistribution", "modification", "restriction",
    "waiver", "audit", "review", "approve", "certify", "validate", "validate license", "validate compliance",
    "update license", "update copyright", "add license", "add copyright", "remove license", "remove copyright",
    "change license", "change copyright"]

list_of_non_functional_code_keywords=["format", "formatting", "indent", "indentation", "style", "styling", "reorganize", "restructure",
    "cleanup", "tidy", "whitespace", "comment", "comments", "annotation", "reorder", "rearrange", "rename",
    "variable rename", "method rename", "class rename", "function rename", "code style", "pep8", "pep257",
    "eslint", "prettier", "checkstyle", "code cleanup", "remove unused", "unused code", "dead code", "simplify",
    "restructure", "split", "merge", "consolidate", "organize imports", "reorganize imports", "lint", "linting",
    "documentation", "docs", "javadoc", "docstring", "header", "footer", "comment block", "add comment",
    "update comment", "remove comment", "readme", "update readme", "fix typo", "typo", "spelling", "grammar",
    "syntax", "coding standards", "coding guidelines", "best practices", "code conventions", "clarify",
    "improve readability", "readability", "consistency", "consistent", "format code", "style guide"]

list_of_keywords_on_meta_data=["datafile", "dataset", "csv", "json", "xml", "yaml", "yml", "config", "configuration", "settings", "properties",
    "env", "environment file", "script", "shell script", "bash", "batch", "Makefile", "CMake", "build script", 
    "build file", "Dockerfile", "docker-compose", "requirements", "dependency", "dependencies", "lock file", 
    "manifest", "metadata", "logfile", "readme", "license", "notebook", "jupyter", "md", 
    "markdown", "rst", "text", "txt", "doc", "docx", "pdf", "image", "jpg", "jpeg", "png", "gif", "svg", "icon", 
    "favicon", "font", "ttf", "otf", "woff", "woff2", "eot", "vector", "diagram", "graph", "chart", "spreadsheet", 
    "xls", "xlsx", "ppt", "pptx", "presentation", "slide", "slides", "archive", "zip", "tar", "gzip", "backup", 
    "restore"
    ]



list_of_module_mgt_keywords=["relocate", "group", "submodule", "subdirectory", "subfolder", "flatten", "hierarchy", "filepath", "filename",
    "filelayout", "directory layout", "folder layout", "module layout", "filesystem", "file organization"]


    
lower_implementation = list(map(str.lower,list_of_implementation_keywords))
lower_maintenance = list(map(str.lower,list_of_maintenance_keywords))
lower_metadata = list(map(str.lower,list_of_keywords_on_meta_data))
lower_nfunctional = list(map(str.lower,list_of_non_functional_code_keywords))
lower_modulemgt = list(map(str.lower,list_of_module_mgt_keywords))
lower_license = list(map(str.lower,list_of_license_keywords))
lower_vcs = list(map(str.lower,list_of_vcs_keywords))

dict_keywords  = {1:lower_implementation,2:lower_maintenance,3:lower_modulemgt,4:lower_license,5:lower_nfunctional,6:lower_vcs,7:lower_metadata}


def generate_csv(distribution_count_dictionary):
    change_type = None
    values_generated = []
    
    if isinstance(distribution_count_dictionary,dict) and bool(distribution_count_dictionary):
        for change_value,count in distribution_count_dictionary.items():
            change_type = classify_changes_type(change_value)

            values_generated.extend([(change_value,change_type)] * count)

        with open("scratch_changes_type_file.csv","w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Change Type","ID"])
            writer.writerows(values_generated)

def generate_cvs_hash_values(distribution_hash_type):
    change_type = None
    commit_sha = None
    values_generated = []
    if isinstance(distribution_hash_type,dict) and bool(distribution_hash_type):
        for commit_sha,values in distribution_hash_type.items():
            if isinstance(values,dict) and bool(values):
                for rev_type,count in values.items():
                    change_type = classify_changes_type(rev_type)
                    values_generated.extend([(commit_sha,rev_type,change_type)] * count)

        with open("scratch_commit_sha_changes_type.csv","w",newline="") as csvcom:
            writer = csv.writer(csvcom)
            writer.writerow(["Commit_Sha","Change Type","ID"])
            writer.writerows(values_generated)




            


def classify_changes_type(change_type_description):
    if isinstance(change_type_description,str):
        if change_type_description == "Implementation":
            return 1
        elif change_type_description == "Maintenance":
            return 2
        elif change_type_description == "Module Management":
            return 3
        elif change_type_description ==  "Legal":
            return 4
        elif change_type_description ==  "Non-functional code":
            return 5
        elif change_type_description == "SCS Management":
            return 6
        elif change_type_description ==  "Meta-Program":
            return 7
        else:
            return -1
            
def plot_changes_type(file_path):
    # Group by 'Change Type' and count the occurrences
    df = pd.read_csv(file_path)
    change_counts = df.groupby('Change Type')['ID'].count().reset_index()
    #change_descr = df['Change ']
    change_counts.columns = ['Change Type', 'Count']

    
    # Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Change Type", y="Count", data=change_counts)
    plt.xticks(rotation=45)
    plt.title('Bar Plot of Scratch3 Change Type Occurences')
    plt.xlabel('Change Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("barplot_scratch3changes2_all.pdf")
    #plt.show()

    '''
    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Change Type", y="Count", data=change_counts)
    plt.xticks(rotation=45)
    plt.title('Box Plot of Scratch3 Change Type Occurence')
    plt.xlabel('Change Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("boxplot_scratch3changes2_all.pdf")
    '''

    # Count Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Change Type", data=df, order=df['Change Type'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Count Plot of Scratch3 Change Type')
    plt.xlabel('Change Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("countplot_scratch3changes2_all.pdf")


def get_parents_from_database(c):
    
    cursor = conn.cursor()
    cursor.execute("SELECT Parent_SHA FROM Commit_Parents WHERE Commit_SHA = (?)", (c,))
    all_parents_of_c = cursor.fetchall()
    cursor.close()
    all_parents_of_c = set([x[0] for x in all_parents_of_c])
    return all_parents_of_c

def check_if_commit_has_parent(c):
    par_comm = get_parents_from_database(c)
    print(type(par_comm))
    return True if len(par_comm) > 0 or par_comm or "None" not in par_comm else False


def decide_implementations(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    
    print(commit_message)
    val = [1 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0  if each_word.lower() in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_vcs and each_word.lower() not in lower_nfunctional]
    return val[0] if len(val) > 0 else -1



def decide_maintenance(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message)
    val = [2 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_maintenance and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_vcs and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_vcs(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message_check)
    val  = [6 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_vcs and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_mod_mgt(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message_check)
    val = [3 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_modulemgt and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_license(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message_check)
    val = [4 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_license and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_non_functional(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message_check)
    val = [5 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_nfunctional and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_license]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_meta_program(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    print(commit_message_check)
    val = [7 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_metadata and each_word.lower() not in lower_implementation and each_word.lower() not in lower_nfunctional and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_license]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_renames(commit_message):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else []
    file_ren = [10 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() == "renames" or each_word.lower() == "rename"]
    return file_ren[0] if len(file_ren) > 0 else -1



def retreive_commit_message(all_project_path,commit_sha,proj_name):
    repo = f'{all_project_path}/{proj_name}' 
    commands = ['git', 'show', '--quiet', '--format=%B', commit_sha.strip()]

    # Check if the repository path exists
    if not os.path.isdir(repo):
        print(f"Error: Repository path {repo} does not exist.")
        return None
    
    try:
        result = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=repo, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        commit_message = result.stdout.strip()
        return commit_message
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None
    #commit_message = subprocess.run(['git show --quiet --format=%B {}'.format(commit_sha.strip())], stdout=subprocess.PIPE, stderr=subprocess.PIPE,cwd=repo, text=True)
    #commit_message = commit_message.stdout.strip()
    #print(type(commit_message))
    #return commit_message       

    
def get_content_parent_sha(c):
    cursor = conn.cursor()
    cursor.execute("SELECT Parent_SHA FROM Commit_Parents WHERE Commit_SHA = (?)", (c,))
    all_parents_of_c = cursor.fetchall()
    cursor.close()
    all_parents_of_c = set([x[0] for x in all_parents_of_c])
    return all_parents_of_c

def decide_which_word_combined(commit_message,dictionary_word):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    word_list = []
    if isinstance(commit_message_check,list) and len(commit_message_check) > 0:
        for each_word in commit_message_check:
            if each_word.lower() in lower_implementation or each_word.lower() in lower_metadata or each_word.lower()  in lower_license or each_word.lower() in lower_maintenance or each_word.lower() in lower_modulemgt or each_word.lower() in lower_vcs or each_word.lower() in lower_nfunctional:
                word_list.extend(use_a_word_determine_where_it_falls(dictionary_word,each_word))
        
        #remove duplicates
        if len(word_list) > 0:
            unique_id = set(word_list)

            #convert to list
            fin_list = list(unique_id)
            return fin_list
        else:
            return []
    else:
        return []
        

def use_a_word_determine_where_it_falls(dictionary_word,word):
    word_id_list = []
    if isinstance(dictionary_word,dict):
        for key,values in dictionary_word.items():
            if isinstance(word,str) and word.lower() in values:
                word_id_list.append(key)
        return word_id_list

def consolidate_algorithm(commit,all_project_path,project_name,dictionary_word):
    change_type_description="Unknown Change"
    commit_message = retreive_commit_message(all_project_path,commit,project_name)
    decid_word = decide_which_word_combined(commit_message,dictionary_word)
    if not isinstance(commit_message,str) or not(commit_message):
        return change_type_description
  

    #check for exclusive values
    if decide_renames(commit_message) != -1 and check_if_commit_has_parent(commit):
        change_type_description = "Maintenance"
        return change_type_description
    
    if decide_implementations(commit_message) != -1:
        change_type_description = "Implementation"
        return change_type_description
    if decide_maintenance(commit_message) != -1:
        change_type_description = "Maintenance"
        return change_type_description
    if decide_meta_program(commit_message) != -1:
        change_type_description = "Meta-Program"
        return change_type_description
    if decide_license(commit_message) != -1:
        change_type_description = "Legal"
        return change_type_description
    if decide_mod_mgt(commit_message) != -1:
        change_type_description = "Module Management"
        return change_type_description
    if decide_non_functional(commit_message) != -1:
        change_type_description = "Non-functional code"
        return change_type_description
    if decide_vcs(commit_message) != -1:
        change_type_description = "SCS Management"
        return change_type_description
    if len(decid_word) > 0:
        if 4 in decid_word:
            change_type_description = "Legal"
        elif 6 in decid_word:
            change_type_description = "SCS Management"
        elif 3 in decid_word:
            change_type_description = "Module Management"
        elif 2 in decid_word and check_if_commit_has_parent(commit):
            change_type_description = "Maintenance"
        elif 1 in decid_word and not check_if_commit_has_parent(commit):
            change_type_description = "Implementation"
        elif 5 in decid_word:
            change_type_description = "Non-functional code"
        elif 7 in decid_word:
            change_type_description = "Meta-Program"
        return change_type_description
    return change_type_description
    
def construct_dictionary(words):
    dict_store = {}
    for each_word in words:
        if each_word not in dict_store:
            dict_store[each_word] = 0
        dict_store[each_word] += 1
    return dict_store


def construct_dictionary_hash(words_hash_types):
    dict_store_hash = {}
    final_store = {}
    if isinstance(words_hash_types,dict):
        for each_hash_type,revis_type in words_hash_types.items():
            if revis_type not in dict_store_hash:
                dict_store_hash[revis_type] = 0
            dict_store_hash[revis_type] += 1
            final_store[each_hash_type] = dict_store_hash
        return final_store 

def integrate_all(all_project_path,dictionary_word,shuffled_data_path):
    chosen_revision_type = "Unknown Change"
    
    all_chosen_type  = []
    with open(shuffled_data_path,"r",encoding="utf-8") as shufd:
        lines = shufd.readlines()
        for each_record in lines:
            content_data = each_record.split(",")
            if len(content_data) == 2:
                project_name = content_data[0]
                commit_sha= content_data[1]
                chosen_revision_type = consolidate_algorithm(commit_sha,all_project_path,project_name,dictionary_word)
                all_chosen_type.append(chosen_revision_type)
                with open("commit_revision_type_id.csv","a") as cri:
                    cri.write(f"{commit_sha.strip()},{chosen_revision_type.strip()},{classify_changes_type(chosen_revision_type.strip())}\n")


def file_has_history(file_path,repo):
    #check if a file has history on GitHub
    result = subprocess.run(['git', 'log', '--', f'{file_path}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=repo)
    
    return result.returncode == 0 and bool(result.stdout.strip())

def file_has_history2(file_path,repo):
    try:
        result = subprocess.run(['git', 'log', '--follow', '--pretty=format:%H', '--', file_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,cwd=repo)
        commits = result.stdout.decode('utf-8').splitlines()
        if len(commits) < 2:
            return False

        if len(commits) > 1:
            for i in range(len(commits) - 1):
                diff_result = subprocess.run(['git', 'diff', '--shortstat', commits[i].strip(), commits[i + 1].strip(), '--', file_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,cwd=repo)
                if diff_result.stdout.strip():
                    return True
            return False
    except subprocess.CalledProcessError:
        return False

def get_commits(filepath,repo):
    result = subprocess.run(['git', 'log', '--follow', '--pretty=format:%H', '--', filepath],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,cwd=repo)
    commits = result.stdout.decode('utf-8').splitlines()
    return commits

def get_file_size(file):
    if os.path.exists(file):
        return os.path.getsize(file)
    return 0

def checkout_commit(commit,file,repo):
    try:
        result = subprocess.run(['git', 'checkout', commit, '--', file], stderr=subprocess.DEVNULL,check=True,cwd=repo)
        return result.returncode
    except subprocess.CalledProcessError as e:
        # Return the returncode on error
        return e.returncode

def checkout_original_branch(repo):
    main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
    main_branch = main_branch.stdout.decode("utf-8") .strip('/n')[0:]
    subprocess.run(['git', 'checkout', main_branch.strip()], stderr=subprocess.DEVNULL,check=True,cwd=repo)

def right_check(all_proj,file_path):
    proj_names = []
    for i in os.listdir(all_proj):
        if len(i) > 1 and os.path.isdir(f'{all_proj}/{i}'):
            i = i.strip() if isinstance(i,str) else i
            proj_names.append(i)
        else:
            continue
    

    prev_size = 0
    commits_that_changed_file = []
    with open(file_path,"r",encoding="utf-8") as fp:
        files = fp.readlines()

        for each_line in files:
            each_line = each_line.strip()
            content = each_line.split(",")

            
            if len(content) == 4:
                proj_name = content[0].strip()
                if proj_name in proj_names:
                    repo = f'{all_proj}/{proj_name}'
                    file_name = content[1].strip()
                    #commit_sha = content[3].strip()

                    all_commits = get_commits(file_name,repo)
                    if len(all_commits) > 1:
                        for each_commit in all_commits:
                            code = checkout_commit(each_commit,file_name,repo)
                            if code == 0:
                                size = get_file_size(file_name)
                                if size != prev_size:
                                    commits_that_changed_file.append(each_commit)
                    
                                prev_size = size
                        print(commits_that_changed_file)
                        checkout_original_branch(repo)
                    
                        if len(commits_that_changed_file) > 1:
                            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/model_deployment/files_that_has_changes.csv","a") as ffcsv:
                                ffcsv.write(f"{each_line}\n")
                        else:
                            continue

                else:
                    continue

def filter_out_non_revision_commits(all_project_path,file_path):
    
    has_revision = False
    with open(file_path,"r",encoding="utf-8") as fp:
        files = fp.readlines()

        for each_line in files:
            each_line = each_line.strip()
            content = each_line.split(",")

            
            if len(content) == 4:
                proj_name = content[0].strip()
                repo = f'{all_project_path}/{proj_name}'
                file_name = content[1].strip()
                commit_sha = content[3].strip()
                #check if the file size increased
                result = subprocess.run(['git', 'cat-file', '-s', f'{commit_sha}:{file_name}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=repo)
                output = result.stdout.strip()
                
                size = int(output) if len(output) > 0 and int(output) != 0  else None
                
                #check if file has history
                if size is None:
                    continue
                else: 
                    has_revision = True
                         
                if has_revision and file_has_history2(file_name,repo):
                    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/model_deployment/filtered_files_2_another.csv","a") as ffcsv:
                        ffcsv.write(f"{each_line}\n")




#filter_out_non_revision_commits("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/model_deployment/filtered_record_proj_name_file_revision_commit.csv")
right_check("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted_test","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/model_deployment/filtered_record_proj_name_file_revision_commit.csv")
#proc = integrate_all("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted",dict_keywords,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/main_project_name_sha_shuffled.csv")
#plot_changes_type("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/scratch_changes_type_file.csv")