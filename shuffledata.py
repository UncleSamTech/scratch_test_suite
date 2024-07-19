import csv
import random



def load_files(path):
    with open(path,mode="r",encoding="utf-8") as shf_file:
        files = list(csv.reader(shf_file))
        
    random.shuffle(files)
        
       
    with open("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/model_deployment/final_shuffled_sample_version2.csv","w") as sgf:
        writer = csv.writer(sgf)
        writer.writerows(files)


def filter_commit_projectname_file(filepath):
    with open(filepath,"r",encoding="utf") as file:
        lines =  file.readlines()

        for each_record in lines:
            content = each_record.split(",")
            if len(content) == 9:
                proj_name = content[0].strip()
                file_name = content[1].strip()
                revision = content[2].strip()
                commit = content[3].strip()

                with open("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/model_deployment/filtered_record_proj_name_file_revision_commit.txt","a") as rec:
                    rec.write(f"{proj_name},{file_name},{revision},{commit}")
                    rec.write(f"\n")

load_files("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/model_deployment/files_that_has_changes.csv")
#filter_commit_projectname_file("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/model_deployment/revisions_record.csv")