#import networkx as nx
import sqlite3
import json
import matplotlib.pyplot as plt


class Scratch_Path:

    def __init__(self):
        self.node = None
        self.root = None
        self.all_hashes = []
        self.all_connections = None
        self.all_paths = []
        self.final_paths = []
        self.all_nodes = None
        self.valid_opcodes = [
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

        self.valid_other_field_codes = ["BACKDROP","DIRECTION","ITEM","OBJECT","STEPS","MESSAGE","CHANGE","OBJECT","SOUND_MENU","VOLUME","TIMES","DISTANCETOMENU","CONDITION"
                                        "OPERAND1","OPERAND2","KEY_OPTION","NUM","INDEX","KEY_OPTION_SPACE","DEGREES","TOWARDS","SECS","SIZE","QUESTION","DX","COSTUME","OPERAND",
                                        "BACKDROP_backdrop1","BROADCAST_INPUT","TOUCHINGOBJECTMENU","WHENGREATERTHANMENU_LOUDNESS_VALUE_10","DURATION","KEY_OPTION_down","KEY_OPTION_up",
                                        "KEY_OPTION_left","KEY_OPTION_right","BROADCAST_OPTION_Game","CONDITION","VALUE","arrow"
                                        ]

    def replace_non_vs_string_with_tokens(self,string_val):
        if isinstance(string_val,str) and len(string_val) > 0:
            val2 = string_val.split()
            
            new_list = ['<S>' if word not in self.valid_opcodes and word not in self.valid_other_field_codes and not word.startswith("BROADCAST_")  else word for word in val2  ]
            
            return " ".join(new_list)
        else:
            return ""



    def get_connection(self):
        conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database6.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor
    
    def get_connection2(self):
        conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_database.db",isolation_level=None)
        cursor =  conn.cursor()
        return conn,cursor

    
    def get_all_contents(self,hash):
        int_val = None
        conn,curr = self.get_connection2()
        if conn != None:
         curr.execute("select distinct(content) from contents where hash = ? ", (hash,))  
         try:
            int_val = curr.fetchall()[0][0]
            #fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
            
         except Exception as e:
             print(e.with_traceback)
         
        return int_val

    def get_all_hashes(self,file_path):
        with open(file_path,"r") as hash:
            all_hashes = hash.readlines()
            if len(all_hashes) > 0:
                for each_hash in all_hashes:
                    each_hash = each_hash.strip()
                    self.all_hashes.append(each_hash)
        return self.all_hashes
    
   
    
    def create_graph(self,all_connections,all_nodes):
        scratch_graph = nx.DiGraph()
        scratch_graph.add_nodes_from(all_nodes)

        for i in range(len(all_connections) - 1):
            scratch_graph.add_edge(all_connections[i],all_connections[i + 1])
        return scratch_graph


    def slice_from_start(self,string_val):
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
            
    def generate_simple_graph(self,file_path,path_name):
        
        hashes = self.get_all_hashes(file_path)
        print(hashes)
            
        if len(hashes) > 0:
            for each_hash in hashes:
                each_hash = each_hash.strip() if isinstance(each_hash,str) else each_hash
                contents = self.get_all_contents(each_hash)
                #print("val",contents)
                if contents is None or len(contents) < 1:
                    continue
                else:
                    contents2 = json.loads(contents)
                    try:
                        self.all_connections = contents2["stats"]["connections"]
                        self.all_nodes = contents2["stats"]["all_nodes"]
                    except:
                        self.all_connections = []
                        self.all_nodes = []
                    if isinstance(self.all_connections,list) and len(self.all_connections) > 0:
                        with open(path_name + each_hash + ".txt","a") as fp:
                            for each_connection in self.all_connections:
                                if each_connection is not None:
                                    try:
                                        val = self.slice_from_start(each_connection)
                                        sec_val  = self.replace_non_vs_string_with_tokens(val)
                                        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/extracted_paths_logs_unique.txt","a") as exp:
                                            exp.write(f"old val {val} replaced value {sec_val}")
                                            exp.write("\n")
                                        
                                    except:
                                        sec_val = ''
                                    if sec_val is None or len(sec_val) < 1:
                                        continue
                                    fp.write(sec_val + " ")
                                    fp.write("\n")
                                else:
                                    continue
                    else: 
                        continue 
      
                
                       
        
    
    def visualize_graph(self,graph):
        sc_gr_pos = nx.spring_layout(graph)
        nx.draw(graph,sc_gr_pos,with_labels=True,arrows=True)
        plt.show()
    
sc_path = Scratch_Path()
#print(sc_path.get_all_hashes("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))
#print(sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/sc_hash_local.txt"))

#sc_path.generate_simple_graph("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/sc_hash_local2_uniq.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/extracted_paths5_unique/")
sc_path.generate_simple_graph("/Users/samueliwuchukwu/documents/scratch_database/scratch_local_hash2.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/sb3_parsed/extracted_paths/")

#v = sc_path.get_all_contents("cfbab365b6dd7f4138823df8ff2e89a108f43dbf8c9950ab27ac8cc981b9adac")
#vis = sc_path.visualize_graph(gr)
#print('contents',v)
#print(vis)