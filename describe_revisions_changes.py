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
    "add functionality", "enhance feature", "extend feature", "introduce feature",
    "feature addition", "feature enhancement", "feature extension", "feature implementation",
    "feature update", "feature change", "feature improvement", "feature modification",
    "feature development", "feature refactor", "feature revision", "feature iteration",
    "feature iteration", "feature upgrade", "feature evolution", "feature innovation",
    "feature enhancement","add", "add file","add project", "Add Scratch3","Add Scratch3","Add"
]

list_of_vcs_keywords=[
    "version", "release", "tag", "snapshot", "update", "changelog", "bump", "merge", "sync", "rebase", "squash",
    "conflict", "revert", "rollback", "history", "commit", "push", "pull", "fetch", "diff", "log", "annotate", 
    "amend", "checkout", "reset", "stage", "unstage", "status", "clean", "clone", "fork", "archive", "export", 
    "import", "save", "recover", "restore", "backup", "stash", "pop", "apply", "init", "set", "integrate",
    "branch", "feature", "hotfix", "develop", "master", "main", "stable", "integration", "checkout", "switch",
    "track", "untrack", "staging", "prod", "env", "pr", "merge request", "cherry-pick", "refactor", "rename",
    "delete", "create", "split", "combine", "pull request", "deploy", "prepare", "migrate", "transition", "promote",
    "demote", "protect", "review", "approve", "reject", "feature toggle",
    "merge branch", "branch update", "branch sync", "version bump", "release branch", "merge conflict",
    "rebase branch", "branch checkout", "branch delete", "branch create", "branch rename", "branch switch",
    "branch track", "branch untrack", "branch fork", "branch clone", "branch clean", "branch save", "branch restore",
    "branch status", "branch protect", "branch review"
]

list_of_maintenance_keywords=[
    "maintenance", "cleanup", "refactor", "tidy", "housekeeping", "reorganize", "restructure", "format", "lint",
    "comment", "documentation", "docs", "update dependencies", "dependency update", "deps", "upgrade", "downgrade",
    "remove", "delete", "deprecate", "obsolete", "cleanup unused", "unused", "rebuild", "configure", "settings",
    "optimize", "improve", "enhance", "speed up", "performance", "perf", "efficient", "reduce", "minimize",
    "maximize", "lightweight", "fast", "speed", "compress", "cache", "tune", "boost", "fine-tune", "profile",
    "debug", "fix", "bug", "issue", "error", "fault", "defect", "glitch", "correct", "resolve", "patch", "handle",
    "address", "troubleshoot", "repair", "solve", "diagnose", "root cause", "log", "trace", "monitor", "alert",
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

list_of_non_functional_code_keywords=["format", "formatting", "indent", "indentation", "style", "styling", "refactor", "reorganize", "restructure",
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
    "manifest", "metadata", "log", "logfile", "readme", "changelog", "license", "notebook", "jupyter", "md", 
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

        with open("changes_type_file.csv","w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Change Type","ID"])
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
    plt.savefig("barplot_scratch3changes2.pdf")
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
    plt.savefig("boxplot_scratch3changes2.pdf")
    '''

    # Count Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Change Type", data=df, order=df['Change Type'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Count Plot of Scratch3 Change Type')
    plt.xlabel('Change Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("countplot_scratch3changes2_.pdf")


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


def decide_implementations(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    
    print(commit_message)
    val = [1 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0  if each_word.lower() in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_vcs and each_word.lower() not in lower_nfunctional]
    return val[0] if len(val) > 0 else -1

def decide_maintenance(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message)
    val = [2 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_maintenance and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_vcs and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_vcs(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message_check)
    val  = [6 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_vcs and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_modulemgt and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_mod_mgt(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message_check)
    val = [3 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_modulemgt and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_license  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_license(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message_check)
    val = [4 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_license and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_nfunctional]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_non_functional(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message_check)
    val = [5 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_nfunctional and each_word.lower() not in lower_implementation and each_word.lower() not in lower_metadata and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_license]
    print(val)
    return val[0] if len(val) > 0 else -1

def decide_meta_program(commit_message,c):
    commit_message_check = commit_message.split() if isinstance(commit_message,str) and len(commit_message) > 0 else ""
    print(commit_message_check)
    val = [7 for each_word in commit_message_check if isinstance(commit_message_check,list) and len(commit_message_check) > 0 if each_word.lower() in lower_metadata and each_word.lower() not in lower_implementation and each_word.lower() not in lower_nfunctional and each_word.lower() not in lower_modulemgt  and each_word.lower() not in lower_vcs and each_word.lower() not in lower_maintenance and each_word.lower() not in lower_license]
    print(val)
    return val[0] if len(val) > 0 else -1



def retreive_commit_message(all_project_path,commit_sha,proj_name):
    repo = f'{all_project_path}/{proj_name}' 
    commands = ['git', 'show', '--quiet', '--format=%B', commit_sha]
    commit_message = subprocess.run(commands, stdout=subprocess.PIPE, cwd=repo, text=True)
    commit_message = commit_message.stdout.strip()
    print(type(commit_message))
    return commit_message       

    
def get_content_parent_sha(c):
    cursor = conn.cursor()
    cursor.execute("SELECT Parent_SHA FROM Commit_Parents WHERE Commit_SHA = (?)", (c,))
    all_parents_of_c = cursor.fetchall()
    cursor.close()
    all_parents_of_c = set([x[0] for x in all_parents_of_c])
    return all_parents_of_c

val = {"Implementation":15,"Maintenance":4,"SCS Management":1}
#generate_csv(val)
#generate_csv({"Implementation":7,"SCS Management":4,"Module Management":4,"Maintenance":5,"Non-functional code":4,"Legal":6})
  
#plot_changes_type("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/changes_type_file.csv")

#decide_implementations("","e80ee0487731d88ef8008e902983b5ac611fe43a")

#commit_message = retreive_commit_message("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/repos","d3c66988e5d92a298b9949c02cf8e926040e4222","chickenclicker")
decide_implementations_val = decide_implementations("Add Scratch","d3c66988e5d92a298b9949c02cf8e926040e4222")
print(decide_implementations_val)
decide_main = decide_maintenance("refactor code","36dh73")
print(decide_main)
decide_vcs_val = decide_vcs("squash","84hjd93")
print(decide_vcs_val)
decide_module_mgt_val = decide_mod_mgt("module layout","5490d0g")
print(decide_module_mgt_val)
dec_lic = decide_license("confidential privacy","e83000f")
print(dec_lic)
dec_nfunct = decide_non_functional("improve readability","938cj4dj")
print(dec_nfunct)
dec_met = decide_meta_program("diagram graph","9030fi")
print(dec_met)

all_parents = get_parents_from_database("709f35a384f7d19bde618e81a91bf57f2372b677")
print(all_parents)
ch = check_if_commit_has_parent('d811ed65d08fef2a3381e32f1e65589efa7b78ab')
print(ch)