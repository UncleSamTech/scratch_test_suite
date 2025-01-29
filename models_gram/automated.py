import os
from xml.etree.ElementPath import find
from sklearn.model_selection import train_test_split
import sqlite3
import json
import random
from pathlib import Path
from itertools import product

valid_opcodes = [
            "event_whenflagclicked",
            "event_whenkeypressed",
            "event_whenthisspriteclicked",
            "event_whenbackdropswitchesto",
            "event_whengreaterthan",
            "event_whenbroadcastreceived",
            "motion_movesteps",
            "motion_turnright",
            "motion_turnleft",
            "motion_goto",
            "motion_goto_menu",
            "motion_gotoxy",
            "motion_glideto",
            "motion_glideto_menu",
            "motion_pointindirection",
            "motion_pointtowards",
            "motion_pointtowards_menu",
            "motion_changexby",
            "motion_setx",
            "motion_changeyby",
            "motion_sety",
            "motion_ifonedgebounce",
            "motion_setrotationstyle",
            "motion_xposition",
            "motion_yposition",
            "motion_direction",
            "looks_sayforsecs",
            "looks_say",
            "looks_thinkforsecs",
            "looks_switchcostumeto",
            "looks_costume",
            "looks_nextcostume",
            "looks_switchbackdropto",
            "looks_backdrops",
            "looks_nextbackdrop",
            "looks_changesizeby",
            "looks_setsizeto",
            "looks_changeeffectby",
            "looks_seteffectto",
            "looks_cleargraphiceffects",
            "looks_show",
            "looks_hide",
            "looks_gotofrontback",
            "looks_goforwardbackwardlayers",
            "looks_costumenumbername",
            "looks_backdropnumbername",
            "looks_size",
            "sound_playuntildone",
            "sound_sounds_menu",
            "sound_play",
            "sound_stopallsounds",
            "sound_changeeffectby",
            "sound_seteffectto",
            "sound_cleareffects",
            "sound_changevolumeby",
            "sound_setvolumeto",
            "sound_volume",
            "event_broadcast",
            "event_broadcastandwait",
            "control_wait",
            "control_repeat",
            "control_forever",
            "control_if",
            "control_if_else",
            "control_wait_until",
            "control_repeat_until",
            "control_stop",
            "control_start_as_clone",
            "control_create_clone_of",
            "control_create_clone_of_menu",
            "control_delete_this_clone",
            "sensing_touchingobject",
            "sensing_touchingobjectmenu",
            "sensing_touchingcolor",
            "sensing_coloristouchingcolor",
            "sensing_distanceto",
            "sensing_distancetomenu",
            "sensing_askandwait",
            "sensing_answer",
            "sensing_keypressed",
            "sensing_keyoptions",
            "sensing_mousedown",
            "sensing_mousex",
            "sensing_mousey",
            "sensing_setdragmode",
            "sensing_loudness",
            "sensing_timer",
            "sensing_resettimer",
            "sensing_of",
            "sensing_of_object_menu",
            "sensing_current",
            "sensing_dayssince2000",
            "sensing_username",
            "operator_add",
            "operator_subtract",
            "operator_multiply",
            "operator_random",
            "operator_gt",
            "operator_lt",
            "operator_equals",
            "operator_and",
            "operator_or",
            "operator_not",
            "operator_join",
            "operator_letter_of",
            "operator_length",
            "operator_contains",
            "looks_think",
            "operator_mod",
            "operator_round",
            "operator_mathop",
            "data_setvariableto",
            "data_changevariableby",
            "data_showvariable",
            "data_hidevariable",
            "data_addtolist",
            "data_deleteoflist",
            "data_deletealloflist",
            "data_insertatlist",
            "data_replaceitemoflist",
            "data_itemoflist",
            "data_itemnumoflist",
            "data_lengthoflist",
            "data_listcontainsitem",
            "data_showlist",
            "data_hidelist",
            "procedures_definition",
            "procedures_prototype",
            "argument_reporter_string_number",
            "argument_reporter_boolean",
            "procedures_call",
            "ThenBlock",
            "BodyBlock",
            "ThenBranch"

        ]

valid_other_field_codes = ["BACKDROP","DIRECTION","ITEM","OBJECT","STEPS","MESSAGE","CHANGE","OBJECT","SOUND_MENU","VOLUME","TIMES","DISTANCETOMENU","CONDITION"
                                        "OPERAND1","OPERAND2","KEY_OPTION","NUM","INDEX","KEY_OPTION_SPACE","DEGREES","TOWARDS","SECS","SIZE","QUESTION","DX","COSTUME","OPERAND",
                                        "BACKDROP_backdrop1","BROADCAST_INPUT","TOUCHINGOBJECTMENU","WHENGREATERTHANMENU_LOUDNESS_VALUE_10","DURATION","KEY_OPTION_down","KEY_OPTION_up",
                                        "KEY_OPTION_left","KEY_OPTION_right","BROADCAST_OPTION_Game","CONDITION","VALUE","arrow","TO"
                                        ]

def get_all_hashes_from_projects(db_path):
    pass


def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_connection2():
        conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_database.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor

def get_all_project_names():
    #make connection
    fin_resp = []
    conn,curs = get_connection()
    RETR_PROJ_QUERY = """SELECT project_name from projects;"""
    if conn != None:
         curs.execute(RETR_PROJ_QUERY)  
         val = curs.fetchall()
         fin_resp = [each_cont[0] for each_cont in val]                   
    else:
        print("connection failed")
    conn.commit()
    
    return fin_resp



def get_all_project_names_opt():
    fin_resp = []
    conn, curs = get_connection()  # Assuming get_connection() handles connection opening
    RETR_PROJ_QUERY = """SELECT project_name FROM projects;"""
    
    try:
        if conn:
            curs.execute(RETR_PROJ_QUERY)
            val = curs.fetchall()
            fin_resp = [each_cont[0] for each_cont in val]  # Extract project names from tuples
        else:
            print("Connection failed")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()  # Close the connection after the operation is done
    
    return fin_resp



def sample_train_test_upd(data, ratio_train, ratio_test):
    if not 0 < ratio_train < 1 or not 0 < ratio_test < 1:
        raise ValueError("ratios must be between 0 and 1")
    
    # Ensure the data is in a list
    data = list(data)
    
    # Shuffle the data to ensure randomness
    random.shuffle(data)
    
    # Calculate the split index based on the train and test ratios
    train_pop_ratio = 0.2 + ratio_train
    total_pop = len(data)
    train_pop = int(train_pop_ratio * total_pop) 

    split_index_train = int(train_pop * ratio_train)  
    split_index_test = int(total_pop * ratio_test)  # Test set will be ratio_test

    # Train set will be from the start to the split index for train
    train_project = data[:split_index_train]
    
    # Test set will be from the end of the train split to the end of the dataset
    test_project = data[split_index_train:split_index_train + split_index_test]

    return train_project, test_project

import random

def sample_train_test_main(data, ratio_train):
    # Validate ratio_train to be between 0 and 1
    if not (0 < ratio_train < 1):
        raise ValueError("ratio_train must be between 0 and 1")

    # Ensure data is a list and shuffle it
    data = list(data)
    random.shuffle(data)

    # Calculate the total size of the data
    total_size = len(data)

    # Test set will always be 20% of the entire population
    test_size = int(0.2 * total_size)

    # Train set will be from 20% to (20% + ratio_train%)
    train_size = int((0.2 + ratio_train) * total_size)

    # The test set is the first 20% of the shuffled data
    test_set = data[:test_size]

    # The train set is from 20% to (20% + ratio_train%) of the shuffled data
    train_set = data[test_size:train_size]

    return train_set, test_set



def sample_train_test_train(data, ratio_train):
    if not 0 < ratio_train < 1:
        raise ValueError("ratio_train must be between 0 and 1")

    data = list(data)  # Ensure it's a list
    random.shuffle(data)  # Shuffle to ensure randomness

    split_index = int(len(data) * ratio_train)  # Compute split point
    return data[:split_index]  # Return the training set

def sample_train_test_test(data, ratio_test, train_project):
    if not 0 < ratio_test < 1:
        raise ValueError("ratio_test must be between 0 and 1")

    data = list(data)  # Ensure it's a list
    random.shuffle(data)  # Shuffle to ensure randomness

    # Remove elements in train_project from data to ensure independence
    test_data = [d for d in data if d not in train_project]

    split_index = int(len(test_data) * ratio_test)  # Compute split point for test set
    test_project = test_data[:split_index]  # Test set

    return test_project

# def sample_train_test_test(data, ratio_test, train_project):
#     test_project, _ = train_test_split([d for d in data if d not in train_project], train_size=ratio_test, random_state=None)
#     return test_project


def retr_hash_match_project(project_name):
    hash_list = []
    conn,curs = get_connection()
    GET_HASHES = """SELECT hash FROM revisions WHERE project_name = ?;"""
    if conn != None:
        curs.execute(GET_HASHES,(project_name,))
        hashes = curs.fetchall()
        hash_list = [each_hash[0] for each_hash in hashes]
    else:
        print("connection failed")
    return hash_list



def retr_hash_match_project_opt(project_name):
    hash_list = []
    conn, curs = get_connection()  # Assuming get_connection() handles connection opening
    GET_HASHES = """SELECT hash FROM revisions WHERE project_name = ?;"""
    
    try:
        if conn:
            curs.execute(GET_HASHES, (project_name,))
            hashes = curs.fetchall()
            hash_list = [each_hash[0] for each_hash in hashes]  # Extract hashes from tuples
        else:
            print("Connection failed")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()  # Close the connection after the operation is done
            
    return hash_list


def retr_all_hash_for_proj_set(all_projects):

    all_hash = []
    if all_projects:
        for each_project in all_projects:
            each_project =  each_project.strip()
            res_hash = retr_hash_match_project(each_project)
            all_hash.extend(res_hash)
    return all_hash

def get_train_hash_unique_test_hash(data, train_ratio):
    # Sample the data into train and test sets
    train_proj, test_proj = sample_train_test_main(data, train_ratio)

    # Initialize sets to hold hashes (to avoid duplicates automatically)
    test_hashes = set()
    train_hashes = set()

    # Collect test hashes, ensuring no duplicates within the test set itself
    for each_project in test_proj:
        for hashs in retr_hash_match_project_opt(each_project.strip()):
            test_hashes.add(hashs)  # Adding automatically avoids duplicates

    # Collect train hashes, ensuring no duplicates within the train set itself
    for each_project in train_proj:
        for hashs in retr_hash_match_project_opt(each_project.strip()):
            train_hashes.add(hashs)  # Adding automatically avoids duplicates

    # Identify unique test hashes that are not in the train set
    uniq_test_hashes = test_hashes - train_hashes

    return list(train_hashes), list(uniq_test_hashes)


def eliminate_duplicates_test_hashes(train_hashes, test_hashes):
    # Convert train_hashes to a set for fast lookup
    train_set = set(train_hashes)
    
    # Filter out hashes from test_hashes that are already in train_hashes
    unique_test_hashes = [hash_val for hash_val in test_hashes if hash_val not in train_set]
    
    return unique_test_hashes
    
def get_all_contents(hash):
        int_val = None
        conn,curr = get_connection()
        if conn != None:
         curr.execute("select distinct(content) from contents where hash = ? ", (hash,))  
         try:
            int_val = curr.fetchall()[0][0]
            #fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
            
         except Exception as e:
             print(e.with_traceback)
         
        return int_val

def get_all_contents_demo(hash):
        int_val = None
        conn,curr = get_connection2()
        if conn != None:
         curr.execute("select distinct(content) from contents where hash = ? ", (hash,))  
         try:
            int_val = curr.fetchall()[0][0]
            #fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
            
         except Exception as e:
             print(e.with_traceback)
         
        return int_val

def replace_non_vs_string_with_tokens(string_val):
        if isinstance(string_val,str) and len(string_val) > 0:
            val2 = string_val.split()
            
            new_list = ['<literal>' if word not in valid_opcodes and word not in valid_other_field_codes and not word.startswith("BROADCAST_")  else word for word in val2  ]
            
            return " ".join(new_list)
        else:
            return ""
        
import re

def replace_non_vs_string_with_tokens_opt(string_val):
    if isinstance(string_val, str) and len(string_val) > 0:
        val2 = string_val.split()

        # Extract all valid prefixes from both lists
        valid_prefixes = {word.split("_")[0] for word in valid_opcodes + valid_other_field_codes if "_" in word}

        # Detect additional prefixes in the input (words containing "_")
        detected_prefixes = {word.split("_")[0] for word in val2 if "_" in word}

        # Merge both sets to cover unknown but similar patterns
        all_prefixes = valid_prefixes.union(detected_prefixes)

        new_list = [
            word if word in valid_opcodes or word in valid_other_field_codes or any(word.startswith(prefix + "_") for prefix in all_prefixes)
            else "<literal>"
            for word in val2
        ]

        return " ".join(new_list)
    else:
        return ""



def slice_from_start(string_val):
        val = ''
        if string_val is not None:
            try:
                val = " ".join(string_val)
            except:
                val = ''
            keywords = ["event_","control_","procedures_"]
            if len(val) > 0:
                start_position = min((val.find(keyword) for keyword in keywords if keyword in val), default=-1)
                if start_position != -1:
                    extr_text = val[start_position:]
            
                    return extr_text
                
all_connections = []
#all_nodes = []


def generate_simple_graph_optimized(path_name,log_path,log_filename,hashes,ngram,run):
        
        #print(hashes)
        # Ensure paths end with a slash
        path_name = path_name if path_name.endswith("/") else path_name + "/"
        log_path = log_path if log_path.endswith("/") else log_path + "/"
        if not hashes:
            return


        # Pre-process hashes by stripping white spaces
        hashes = [h.strip() for h in hashes if isinstance(h, str)]

        # Open the extracted paths log file once
        with open(f"{log_path}/{log_filename}_{ngram}_{run}.txt", "a") as exp:
            for each_hash in hashes:
                contents = get_all_contents(each_hash)
            
                if not contents:  # Continue if no contents
                    continue

                # Attempt to parse the contents
                try:
                    contents2 = json.loads(contents)
                    all_connections = contents2["stats"].get("connections", [])
                    #all_nodes = contents2["stats"].get("all_nodes", [])
                except (KeyError, ValueError):
                    all_connections = []
                    #all_nodes = []

                # Skip if there are no connections
                if not isinstance(all_connections, list) or not all_connections:
                    continue

                # Open the output file for this hash once
                with open(f"{path_name}{each_hash}.txt", "a") as fp:
                    for each_connection in all_connections:
                        if not each_connection:
                            continue

                        try:
                            val = slice_from_start(each_connection)
                            sec_val = replace_non_vs_string_with_tokens_opt(val)

                            # Log the old and replaced values
                            exp.write(f"old val {val} replaced value {sec_val}\n")
                        except Exception:
                            sec_val = ''

                        # Continue if the processed value is empty
                        if not sec_val:
                            continue

                        # Write the processed value to the output file
                        fp.write(f"{sec_val}\n")


def generate_simple_graph_optimized2(path_name, log_path, log_filename, hashes, ngram, run):
    if not hashes:
        return

    path_name = str(path_name) if str(path_name).endswith("/") else str(path_name) + "/"
    log_path = str(log_path) if str(log_path).endswith("/") else str(log_path) + "/"

    original_dir = os.getcwd()

    try:
        os.chdir(path_name)
        hashes = [h.strip() for h in hashes if isinstance(h, str)]

        os.chdir(log_path)
        log_file_path = f"{log_filename}_{ngram}_{run}.txt"

        # Open log file once
        with open(log_file_path, "a") as exp:
            os.chdir(path_name)

            for each_hash in hashes:
                contents = get_all_contents(each_hash)

                if not contents:
                    continue

                try:
                    contents2 = json.loads(contents)
                    all_connections = contents2["stats"].get("connections", [])
                except (KeyError, ValueError):
                    all_connections = []

                if not isinstance(all_connections, list) or not all_connections:
                    continue

                hash_file_path = f"{each_hash}.txt"

                # **Track what we write in this execution only**
                written_this_run = set()

                with open(hash_file_path, "a") as fp:
                    for each_connection in all_connections:
                        if not each_connection:
                            continue

                        try:
                            val = slice_from_start(each_connection)
                            sec_val = replace_non_vs_string_with_tokens_opt(val)
                            exp.write(f"old val {val} replaced value {sec_val}\n")
                        except Exception:
                            sec_val = ''

                        # **Ensure we don't write the same content twice in this execution**
                        if not sec_val or sec_val in written_this_run:
                            continue

                        fp.write(f"{sec_val}\n")
                        written_this_run.add(sec_val)  # Track what we've written this run

    finally:
        os.chdir(original_dir)  # Restore original directory
            

def generate_simple_graph_optimized2_demo(path_name, log_path, log_filename, hashes):
    if not hashes:
        return

    path_name = str(path_name) if str(path_name).endswith("/") else str(path_name) + "/"
    log_path = str(log_path) if str(log_path).endswith("/") else str(log_path) + "/"

    original_dir = os.getcwd()

    try:
        os.chdir(path_name)
        hashes = [h.strip() for h in hashes if isinstance(h, str)]

        os.chdir(log_path)
        log_file_path = f"{log_filename}.txt"

        # Open log file once
        with open(log_file_path, "a") as exp:
            os.chdir(path_name)

            for each_hash in hashes:
                contents = get_all_contents_demo(each_hash)

                if not contents:
                    continue

                try:
                    contents2 = json.loads(contents)
                    all_connections = contents2["stats"].get("connections", [])
                except (KeyError, ValueError):
                    all_connections = []

                if not isinstance(all_connections, list) or not all_connections:
                    continue

                hash_file_path = f"{each_hash}.txt"

                # **Track what we write in this execution only**
                written_this_run = set()

                with open(hash_file_path, "a") as fp:
                    for each_connection in all_connections:
                        if not each_connection:
                            continue

                        try:
                            val = slice_from_start(each_connection)
                            # sec_val = replace_non_vs_string_with_tokens(val)
                            exp.write(f"old val {val} replaced value {sec_val}\n")
                        except Exception:
                            sec_val = ''

                        # **Ensure we don't write the same content twice in this execution**
                        if not val or val in written_this_run:
                            continue

                        fp.write(f"{val}\n")
                        written_this_run.add(val)  # Track what we've written this run

    finally:
        os.chdir(original_dir)  # Restore original directory



tr_hash = ["ca636d33bc2247d4e9459cdb0a7d3a2ad813fbd3ebbadf3d30535250e03b21db", "6eb0fa5c8923c2b2f2e31526ee5f42b6efae017f9c9d08c1453d8be872ea968c","ca636d33bc2247d4e9459cdb0a7d3a2ad813fbd3ebbadf3d30535250e03b21db"]
ts_hash = ["caef21eeee612e6aebae1b6181779ecebc37214ff3148801b9ec899059330416","ca636d33bc2247d4e9459cdb0a7d3a2ad813fbd3ebbadf3d30535250e03b21db","ca636d33bc2247d4e9459cdb0a7d3a2ad813fbd3ebbadf3d30535250e03b21db"]

def generate_paths(base_path, models, train_hashes, test_hashes):
    for each_model, each_gram, each_run in product(models, range(2,7), range(1,6)):  
        train_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}/")
        test_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}_test/")
        log_dir = Path(f"{base_path}{each_model}/path_{each_model}_logs")
        log_dir_test = Path(f"{base_path}{each_model}/path_{each_model}_logs_test")

        # Ensure directories exist
        for dir in [train_dir, test_dir, log_dir, log_dir_test]:
            dir.mkdir(exist_ok=True)

        print(f"train_dir {train_dir}")
        print(f"test_dir {test_dir}")

        # Ensure directories exist before proceeding
        if all([train_dir.exists(), test_dir.exists(), log_dir.exists(), log_dir_test.exists()]):
            # Shuffle without duplication concerns
            shuffled_train_hash = random.sample(set(train_hashes), len(set(train_hashes)))
            shuffled_test_hash = random.sample(set(test_hashes), len(set(test_hashes)))

            # Ensure test hashes are unique from train hashes
            unique_test_hashes = eliminate_duplicates_test_hashes(shuffled_train_hash, shuffled_test_hash)

            generate_simple_graph_optimized2(train_dir, log_dir, "logs_test", shuffled_train_hash, each_gram, each_run)

            generate_simple_graph_optimized2(test_dir, log_dir_test, "logs_test", unique_test_hashes, each_gram, each_run)

def generate_paths2(train_hashes, test_hashes,train_dir,test_dir,log_dir,log_dir_test):
   
        shuffled_train_hash = sorted(set(train_hashes))
        shuffled_test_hash = sorted(set(test_hashes))

        # Ensure test hashes are unique from train hashes
        unique_test_hashes = eliminate_duplicates_test_hashes(shuffled_train_hash, shuffled_test_hash)

        generate_simple_graph_optimized2_demo(train_dir, log_dir, "logs_test", shuffled_train_hash)

        generate_simple_graph_optimized2_demo(test_dir, log_dir_test, "logs_test", unique_test_hashes)



def generate_paths_opt(base_path, models, ratio_split):
    all_projects = get_all_project_names_opt()
    for each_model, each_gram, each_run in product(models, range(2,7), range(1,6)):  
        train_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}/")
        test_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}_test/")
        log_dir = Path(f"{base_path}{each_model}/path_{each_model}_logs")
        log_dir_test = Path(f"{base_path}{each_model}/path_{each_model}_logs_test")

        
        train_hashes, test_hashes = get_train_hash_unique_test_hash(all_projects,ratio_split)

        # Ensure directories exist
        for dir in [train_dir, test_dir, log_dir, log_dir_test]:
            dir.mkdir(exist_ok=True)

        print(f"train_dir {train_dir}")
        print(f"test_dir {test_dir}")

        # Ensure directories exist before proceeding
        if all([train_dir.exists(), test_dir.exists(), log_dir.exists(), log_dir_test.exists()]):
            generate_simple_graph_optimized2(train_dir, log_dir, "logs_test", train_hashes, each_gram, each_run)

            generate_simple_graph_optimized2(test_dir, log_dir_test, "logs_test", test_hashes, each_gram, each_run)



#generate_paths2(tr_hash,ts_hash,"/Users/samueliwuchukwu/desktop/auto/train","/Users/samueliwuchukwu/desktop/auto/test","/Users/samueliwuchukwu/desktop/auto/log","/Users/samueliwuchukwu/desktop/auto/log_test")


generate_paths_opt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[10],0.1)
generate_paths_opt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[20],0.2)
generate_paths_opt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[30],0.3)
generate_paths_opt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[50],0.5)
generate_paths_opt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[80],0.8)






# generate_paths("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[50],retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.5)),eliminate_duplicates_test_hashes(retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.5)),retr_all_hash_for_proj_set(sample_train_test_test(get_all_project_names(),0.2))))
# generate_paths("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[80],retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.8)),eliminate_duplicates_test_hashes(retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.8)),retr_all_hash_for_proj_set(sample_train_test_test(get_all_project_names(),0.2))))

# test_path_20_o6_r1= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_1_test/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs_test","logs_test",uniq_test_hashes,6,1)
# train_path_20_o6_r1= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_1/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs","logs",train_hashes,6,1)
# test_path_20_o6_r2= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_2_test/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs_test","logs_test",uniq_test_hashes,6,2)
# train_path_20_o6_r2= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_2/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs","logs",train_hashes,6,2)
# test_path_20_o6_r3= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_3_test/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs_test","logs_test",uniq_test_hashes,6,3)
# train_path_20_o6_r3= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_3/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs","logs",train_hashes,6,3)
# test_path_20_o6_r4= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_4_test/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs_test","logs_test",uniq_test_hashes,6,4)
# train_path_20_o6_r4= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_4/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs","logs",train_hashes,6,4)
# test_path_20_o6_r5= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_5_test/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs_test","logs_test",uniq_test_hashes,6,5)
# train_path_20_o6_r5= generate_simple_graph_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_6_5/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_logs","logs",train_hashes,6,5)


#comm -12 <(ls "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_2_2" | sort) <(ls "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_2_2_test" | sort) | xargs -I {} rm "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20/path_20_2_2_test/{}"


#!/bin/bash

# Base directory
# BASE_DIR="/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20"

# # Loop through n-gram orders (2 to 6)
# for ngram in {2..6}; do
#   # Loop through runs (1 to 5)
#   for run in {1..5}; do
#     # Define train and test directories
#     TRAIN_DIR="$BASE_DIR/path_20_${ngram}_${run}"
#     TEST_DIR="${TRAIN_DIR}_test"

#     # Check if both directories exist
#     if [[ -d "$TRAIN_DIR" && -d "$TEST_DIR" ]]; then
#       echo "Processing: $TRAIN_DIR and $TEST_DIR"

#       # Find duplicate files in test directory and delete them
#       comm -12 <(ls "$TRAIN_DIR" | sort) <(ls "$TEST_DIR" | sort) | xargs -I {} rm "$TEST_DIR/{}"

#       echo "Deleted duplicates from $TEST_DIR"
#     else
#       echo "Skipping: $TRAIN_DIR or $TEST_DIR does not exist."
#     fi
#   done
# done

#for n in {2..6}; do for r in {1..5}; do TRAIN="/media/crouton/siwuchuk/newdir/vscode_repos_files/method/80/path_80_${n}_${r}"; TEST="${TRAIN}_test"; [[ -d "$TRAIN" && -d "$TEST" ]] && comm -12 <(ls "$TRAIN" | sort) <(ls "$TEST" | sort) | xargs -I {} rm "$TEST/{}"; done; done
# find /media/crouton/siwuchuk/newdir/vscode_repos_files/method/10 -mindepth 1 -delete
# find /media/crouton/siwuchuk/newdir/vscode_repos_files/method/20 -mindepth 1 -delete
# find /media/crouton/siwuchuk/newdir/vscode_repos_files/method/30 -mindepth 1 -delete
# find /media/crouton/siwuchuk/newdir/vscode_repos_files/method/50 -mindepth 1 -delete
# find /media/crouton/siwuchuk/newdir/vscode_repos_files/method/80 -mindepth 1 -delete