import csv
import random



def load_files(path):
    with open(path,mode="r",encoding="utf-8") as shf_file:
        files = list(csv.reader(shf_file))
        
    random.shuffle(files)
        
       
    with open("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/main_project_name_sha_shuffled.csv","w") as sgf:
        writer = csv.writer(sgf)
        writer.writerows(files)


load_files("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/project_name_commit_sha_revisions_types.csv")