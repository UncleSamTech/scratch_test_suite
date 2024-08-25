import csv
import random



def load_files(path):
    with open(path,mode="r",encoding="utf-8") as shf_file:
        files = list(csv.reader(shf_file))
        
    random.shuffle(files)
        
   
    with open("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/model_deployment/new_shuffled_record.csv","w") as sgf:
        writer = csv.writer(sgf)
        writer.writerows(files)


def remove_redundant_records(file_path,output_file_path):
    unique_entries = set()

    with open(file_path,"r",encoding="utf-8") as inp_file:
        lines = inp_file.readlines()

    with open(output_file_path,"w") as out_file:
        for line in lines:
            contents = line.strip().split(',')
            if len(contents) < 9:
                continue
            project_name = contents[0].strip()
            revision_name = contents[2].strip()
            commit_sha = contents[3].strip()
            hash_value = contents[5].strip()
            entry_key = (project_name,revision_name,commit_sha,hash_value)

            if entry_key not in unique_entries:
                unique_entries.add(entry_key)
                out_file.write(line)
            else:
                with open("filteredoutduplicatefiles.csv","a") as fil:
                    fil.write(line)



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

load_files("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/main_revisions_shuffled_output.csv")
#filter_commit_projectname_file("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/model_deployment/revisions_record.csv")

#remove_redundant_records("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/main_revisions_shuffled.csv","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/main_revisions_shuffled_output.csv")