import os
from sklearn.model_selection import train_test_split
import sqlite3
import json
import random
from pathlib import Path

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
                                        "KEY_OPTION_left","KEY_OPTION_right","BROADCAST_OPTION_Game","CONDITION","VALUE","arrow"
                                        ]

def get_all_hashes_from_projects(db_path):
    pass


def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db",isolation_level=None)
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

def sample_train_test_train(data, ratio_train):
    train_project,_ = train_test_split(data, train_size=ratio_train, random_state=None)   
    return train_project

def sample_train_test_test(data, ratio_test):
    test_project, _ = train_test_split(data, train_size=ratio_test, random_state=None)
    return test_project

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

def retr_all_hash_for_proj_set(all_projects):
    all_hash = []
    if all_projects:
        for each_project in all_projects:
            each_project =  each_project.strip()
            res_hash = retr_hash_match_project(each_project)
            all_hash.extend(res_hash)
    return all_hash

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

def replace_non_vs_string_with_tokens(string_val):
        if isinstance(string_val,str) and len(string_val) > 0:
            val2 = string_val.split()
            
            new_list = ['<literal>' if word not in valid_opcodes and word not in valid_other_field_codes and not word.startswith("BROADCAST_")  else word for word in val2  ]
            
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
                            sec_val = replace_non_vs_string_with_tokens(val)

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

    # Convert PosixPath to string and ensure paths end with a slash
    path_name = str(path_name) if str(path_name).endswith("/") else str(path_name) + "/"
    log_path = str(log_path) if str(log_path).endswith("/") else str(log_path) + "/"

    # Save the original working directory
    original_dir = os.getcwd()

    try:
        # Change to the `path_name` directory for writing hash files
        os.chdir(path_name)

        # Pre-process hashes by stripping white spaces
        hashes = [h.strip() for h in hashes if isinstance(h, str)]

        # Change to the `log_path` directory for writing log files
        os.chdir(log_path)

        # Open the extracted paths log file once
        with open(f"{log_filename}_{ngram}_{run}.txt", "a") as exp:
            # Change back to the `path_name` directory for writing hash files
            os.chdir(path_name)

            for each_hash in hashes:
                contents = get_all_contents(each_hash)

                if not contents:  # Continue if no contents
                    continue

                # Attempt to parse the contents
                try:
                    contents2 = json.loads(contents)
                    all_connections = contents2["stats"].get("connections", [])
                except (KeyError, ValueError):
                    all_connections = []

                # Skip if there are no connections
                if not isinstance(all_connections, list) or not all_connections:
                    continue

                # Open the output file for this hash once
                with open(f"{each_hash}.txt", "a") as fp:  # No need to include `path_name` in the file path
                    for each_connection in all_connections:
                        if not each_connection:
                            continue

                        try:
                            val = slice_from_start(each_connection)
                            sec_val = replace_non_vs_string_with_tokens(val)

                            # Log the old and replaced values
                            exp.write(f"old val {val} replaced value {sec_val}\n")
                        except Exception:
                            sec_val = ''

                        # Continue if the processed value is empty
                        if not sec_val:
                            continue

                        # Write the processed value to the output file
                        fp.write(f"{sec_val}\n")

    finally:
        # Restore the original working directory
        os.chdir(original_dir)
            

train_splits = [0.2,0.3,0.5,0.8]
model_numbers = [20]


def generate_paths(base_path,models,train_hashes,test_hashes):
#
    for each_model in models:
        for each_gram in range(2,7):
            for each_run in range(1,6):
                train_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}/")
                test_dir = Path(f"{base_path}{each_model}/path_{each_model}_{each_gram}_{each_run}_test/")
                log_dir = Path(f"{base_path}{each_model}/path_{each_model}_logs")
                log_dir_test = Path(f"{base_path}{each_model}/path_{each_model}_logs_test")
                train_dir.mkdir(exist_ok=True)
                test_dir.mkdir(exist_ok=True)
                log_dir.mkdir(exist_ok=True)
                log_dir_test.mkdir(exist_ok=True)
                print(f"train_dir {train_dir}")
                print(f"test_dir {test_dir}")
                if train_dir.exists() and test_dir.exists() and log_dir.exists() and log_dir_test.exists():
                    # shuffled_train_hash = random.sample(train_hashes, len(train_hashes))  # Shuffle train hashes
                    # shuffled_test_hash = random.sample(test_hashes, len(test_hashes))  # Shuffle test hashes

                    # Compare shuffled test hashes with shuffled train hashes
                    #unique_test_hashes = eliminate_duplicates_test_hashes(shuffled_train_hash, shuffled_test_hash)

                    # Generate graphs for shuffled train and unique test hashes
                    generate_simple_graph_optimized2(train_dir, log_dir, "logs_test", train_hashes, each_gram, each_run)
                    generate_simple_graph_optimized2(test_dir, log_dir_test, "logs_test", test_hashes, each_gram, each_run)


# train_hash = retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.1))
# test_hash = retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.2))
# uniq_hash = eliminate_duplicates_test_hashes(train_hash,test_hash)
# generate_paths("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[10],train_hash,uniq_hash)



generate_paths("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/",[10],retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.1)),eliminate_duplicates_test_hashes(retr_all_hash_for_proj_set(sample_train_test_train(get_all_project_names(),0.1)),retr_all_hash_for_proj_set(sample_train_test_test(get_all_project_names(),0.2))))

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
