import os
import json
import collections
from unzip_scratch import unzip_scratch

class scratch_parser:

    def __init__(self):
        
        self.blocs_json = None
        self.blocks_values = []
        self.sb3class = unzip_scratch()
        self.ommited_block_keys_parent = {"opcode"}
        self.all_opcodes = []
        self.scratch_tree_list = []
        self.scratch_stats = {}
        self.next_val_tree = {}
        self.input_block = {}
        self.sec_val = None
        self.in_val = None
        self.new_parent_tree_met = {}
        self.all_met = []
        self.inpt_2 = []
        self.missed_inp  = []
        self.missed_inp2  = []
        self.child_input_keys = []

    
    def get_all_targets(self,json_data):
        if isinstance(json_data,dict) and bool(json_data):
            return json_data["targets"] if 'targets' in json_data.keys() else {}
        
    
    def get_all_blocks_vals(self,blocks_values):
        targ = self.get_all_targets(blocks_values)
        return {'blocks':each_block['blocks'] for each_block in targ if isinstance(each_block,dict) and 'blocks' in each_block.keys()}
    
    def get_only_blocks(self,block_targ):
        if block_targ == None or block_targ == {}:
            return {}
        all_blocks =  self.get_all_blocks_vals(block_targ)
        return all_blocks['blocks']
    
    def get_any_block_by_id(self,blocks_values,key):
        if key == None or key == '' or blocks_values == None or blocks_values == {} or blocks_values['blocks'] == None or blocks_values['blocks'] == {} or blocks_values['blocks'][key] == None or blocks_values['blocks'][key] == {}:
            return {}
        return blocks_values['blocks'][key]
        

    def get_opcode_from_id(self,block_values,block_id):
        if block_id == None or block_id == '':
            return ''
        if block_values['blocks'][block_id]['fields'] == {} or block_values['blocks'][block_id]['fields'] == None:
            return block_values['blocks'][block_id]['opcode'] if block_values['blocks'][block_id]['opcode'] != None else ''
        else:
            opcode = block_values["blocks"][block_id]["opcode"]
            fields = block_values["blocks"][block_id]["fields"] 
            if isinstance(fields,dict) and bool(fields):
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                                opcode = f'{opcode}_{v[0]}'
                        elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and v[1] != None and len(v[1]) > 0:
                            opcode = f'{opcode}_{v[0]}_{v[1]}'
            return opcode
        
    def return_all_opcodes(self,blocks_values):
        return [self.get_opcode_from_id(blocks_values,k2) for k,v in blocks_values.items() for k2,v2 in v.items() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2)]
    
    def get_all_unique_opcodes(self,blocks_values):
        all_unique_opcodes = []
        if blocks_values == None or blocks_values == {}:
            return []
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2 in v.keys():
                        opcodes = self.get_opcode_from_id(blocks_values,k2)
                        if opcodes not in all_unique_opcodes:
                            all_unique_opcodes.append(opcodes)
                        else:
                            continue

        return all_unique_opcodes
        


    def get_parent_opcode(self,blocks_values):
        if blocks_values == None or blocks_values == {}:
            return ''
        par = [v2['opcode'] for k,v in blocks_values.items() for v2 in v.values() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2) and 'opcode' in v2.keys() and 'parent' in v2.keys() and v2["parent"] == None]
        return par[0] if len(par) == 1 else par
         
    
    def read_input_values_by_id(self,blocks_values,id):
        if id == None or id == '' or blocks_values == None or blocks_values == {}:
            return {}
        if isinstance(blocks_values,dict) and bool(blocks_values) and isinstance(blocks_values['blocks'],dict) and bool(blocks_values['blocks']):
            block = blocks_values['blocks']
            
            if isinstance(block,dict) and bool(block) and id in block.keys():
                block_val = block[id]
                
                if isinstance(block_val,dict) and bool(block_val) and 'inputs' in block_val.keys():
                    return block_val['inputs']
            #return blocks_values['blocks'][id]['inputs'] if 'inputs' in blocks_values['blocks'][id].keys() else {}
    
    def check_dict_depth(self,dict_val,depth=1):
        if not isinstance(dict_val,dict) or not bool(dict_val):
            return depth
        return max(self.check_dict_depth(v,depth+1) for k,v in dict_val.items())

    
    def get_children_key_recursively(self,blocks_values,spec_block):
        if spec_block == None or spec_block == {} or blocks_values == None or blocks_values == {}:
            return []
        
        inp_block = spec_block["inputs"] if "inputs" in spec_block.keys() else {}
        if isinstance(inp_block,dict) and bool(inp_block):
            for k,v in inp_block.items():
                if isinstance(v,list) and len(v) > 0:
                    for each_val in v:
                        if isinstance(each_val,str):
                            if len(each_val) > 0:
                                self.child_input_keys.append(each_val)
                                bloc = self.get_any_block_by_id(blocks_values,each_val)
                                if bloc["inputs"] != None or bloc["inputs"] != {}:
                                    self.get_children_key_recursively(blocks_values,bloc) 
                                else:
                                    break                
        return self.child_input_keys
    
    
    def get_next_child_keys(self,blocks_values,inp_block):
        all_next_keys = []
        all_child_keys = self.get_children_key_recursively(blocks_values,inp_block)
        for each_key in all_child_keys:
            block = self.get_any_block_by_id(blocks_values,each_key)
            if isinstance(block,dict) and bool(block) and 'next' in block.keys():
                all_next_keys.append(block['next'])
        return all_next_keys

    def get_all_parent_keys(self,blocks_values):
        all_parent_keys = []
        if blocks_values == None or blocks_values == {}:
            return []
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2,v2 in v.items():
                        if isinstance(v2,dict) and bool(v2) and 'parent' in v2.keys() and v2['parent'] == None:
                            all_parent_keys.append(k2)
        return all_parent_keys
    
    def compare_parent_keys(self,blocks_values,block_key,parent_key):
        if blocks_values == None or blocks_values == {} or block_key == None or block_key == {} or parent_key == None or parent_key == '':
            return False
        
        if isinstance(block_key,dict) and bool(block_key) and 'parent' in block_key.keys():
            parent_block = self.get_any_block_by_id(blocks_values,block_key['parent'])
            if block_key['parent'] != None and block_key['parent'] == parent_key:
                return True
            
            else:
                next_par = self.compare_parent_keys(blocks_values,parent_block,parent_key)
                return next_par
        

    def break_down(self,blocks_values,parent_key):
        spec = []
        if blocks_values == None or blocks_values == {} or parent_key == None or parent_key == '':
            return []
        for k,v in blocks_values.items():
            if isinstance(v,dict) and bool(v):
                for k2,v2 in v.items():
                    if isinstance(v2,dict) and bool(v2):
                        if parent_key in self.get_all_parent_keys(blocks_values) and v2["next"] not in self.get_children_key_recursively(blocks_values,self.get_any_block_by_id(blocks_values,k2)) and v2["next"] not in self.get_next_child_keys(blocks_values,self.get_any_block_by_id(blocks_values,k2)) and self.compare_parent_keys(blocks_values,self.get_any_block_by_id(blocks_values,v2["next"]),parent_key):
                            spec.append(v2["next"])
        return spec
                        

    def get_all_next_id_test(self,blocks_values):
       if blocks_values == None or blocks_values == {}:
            return {}                                                   
       return {self.get_opcode_from_id(blocks_values,each_value):self.break_down(blocks_values,each_value) for each_value in self.get_all_parent_keys(blocks_values)}

    def get_input_block_by_id_key(self,block_values,bid,key):
        if key == None or len(key) < 1 or block_values == None or block_values == {} or bid == None or len(bid) < 1:
            return []
        specific_input_by_id_key = []
        input_block = self.read_input_values_by_id(block_values,bid)
        if isinstance(input_block,dict) and bool(input_block):
            if key in input_block.keys():
                value_block =  input_block[key]
                if isinstance(value_block,list) and len(value_block) > 0:
                    for each_val in value_block:
                        if isinstance(each_val,str) and len(each_val) > 0:
                            opcode = self.get_opcode_from_id(block_values,each_val)
                            specific_input_by_id_key = [key,[opcode]]
                        elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                            specific_input_by_id_key = [key,[each_val[1]]]
                        


        
        return specific_input_by_id_key

    def correct_input_block_tree_by_id(self,blocks_values,input_block,ids):

        corr_block_tree = []
        if input_block == None or input_block == {} or blocks_values == None or blocks_values == {}:
            return []
        if isinstance(input_block,dict) and bool(input_block):
            for k,v in input_block.items():
                if isinstance(v,list) and len(v) > 0:
                    if isinstance(v[1] ,str) and len(v[1]) > 0:
                        opcode = self.get_opcode_from_id(blocks_values,v[1])  
                        recur_val = self.correct_input_block_tree_by_id(blocks_values,self.read_input_values_by_id(blocks_values,v[1]),v[1])  
                        
                        any_block = self.get_any_block_by_id(blocks_values,v[1])
                        next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                        next_rec  = self.correct_input_block_tree_by_id(blocks_values,self.read_input_values_by_id(blocks_values,any_block["next"]),any_block["next"]) 
                        if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                            corr_block_tree.append([k,[opcode,[recur_val],next_opcode,[next_rec]]])
                        elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                            corr_block_tree.append([k,[opcode,[recur_val]]])
                        elif any_block["next"] == None and next_rec == [] or next_rec == None:
                            corr_block_tree.append([k,opcode])
                    elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                        corr_block_tree.append(self.get_input_block_by_id_key(blocks_values,ids,k))
        return corr_block_tree
        
    def create_next_values2(self,blocks_values):  
        tr = [] 
        
       
        all_val = self.get_all_next_id_test(blocks_values)     
        if all_val == None or all_val == {}:
            return []
        if isinstance(all_val,dict) and bool(all_val):
            for ks,vs in all_val.items():
                if isinstance(vs,list) and len(vs) > 0:
                    val =  [[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                    tr.append([ks,val])
        return tr

    def count_opcodes(self,blocks_values):
        all_opcodes = self.return_all_opcodes(blocks_values)
        if blocks_values == None or blocks_values == {}:
           return {}
           
        if all_opcodes == None or all_opcodes == []:
               return {}
           
        count_val = collections.Counter(all_opcodes)
        return count_val 

    def iterate_tree_for_non_opcodes(self,scratch_tree,blocks_values):
        if scratch_tree == None or scratch_tree == [] or blocks_values == None or blocks_values == {}:
            return []   
        if isinstance(scratch_tree,list) and len(scratch_tree) > 0:
            for each_val in scratch_tree:
                if isinstance(each_val,list) and len(each_val) > 0:
                    self.iterate_tree_for_non_opcodes(each_val,blocks_values)
                else:
                    if each_val not in self.get_all_unique_opcodes(blocks_values) and  each_val != []:
                        self.all_met.append(each_val)
                    
     
        return self.all_met
    
    def count_non_opcodes(self,blocks_values,scratch_tree):
        non_opcodes = self.iterate_tree_for_non_opcodes(scratch_tree,blocks_values)
        if blocks_values == None or blocks_values == {} or scratch_tree == None or scratch_tree == [] or non_opcodes == None or non_opcodes == []:
           return {}
           
        count_val = collections.Counter(non_opcodes)
        return count_val
    
  
    

    
    def generate_summary_stats(self,blocks_values,file_name,scratch_tree):
        opcodes = self.count_opcodes(blocks_values)
        non_opcodes = self.count_non_opcodes(blocks_values,scratch_tree)
        opcode_tree = {}
        non_opcode_tree = {}
        most_common_opcode_tree = {}
        most_common_non_opcode_tree = {}
        opcode_key = None
        opcode_val = None
        non_opcode_key = None
        non_opcode_val = None
        for k in opcodes:
            opcode_key = k
            opcode_val = opcodes[k]
            opcode_tree[opcode_key] = opcode_val
        for mc in non_opcodes:
            non_opcode_key = mc
            non_opcode_val = non_opcodes[mc]
            non_opcode_tree[non_opcode_key] = non_opcode_val
        
        for mc in opcodes.most_common(5):
            most_common_opcode_key = mc[0]
            most_common_opcode_val = mc[1]
            most_common_opcode_tree[most_common_opcode_key] = most_common_opcode_val
        
        for nmc in non_opcodes.most_common(5):
            most_common_non_opcode_key = nmc[0]
            most_common_non_opcode_val = nmc[1]
            most_common_non_opcode_tree[most_common_non_opcode_key] = most_common_non_opcode_val

            
        self.scratch_stats[file_name] = {"opcodes_statistics":opcode_tree,"non_opcodes_statistics":non_opcode_tree,"most_common_opcodes_statistics":most_common_opcode_tree,"most_common_non_opcodes_statistics":most_common_non_opcode_tree}
        return self.scratch_stats 
        



    def read_files(self, parsed_file):
        self.parsed_value = self.sb3class.unpack_sb3(parsed_file)
        self.blocs_json = json.loads(self.parsed_value)
        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
          
        #all opcodes
        #print(self.get_all_unique_opcodes(all_blocks_value))

        #print(self.count_opcodes(all_blocks_value))

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2(all_blocks_value)
        
       

        with open(f"files/{file_name}_tree2.json","w") as tree_file:
            json.dump(next_val2,tree_file,indent=4)
        
        with open(f"files/{file_name}_stats.json","w") as stats_file:
            json.dump(self.generate_summary_stats(all_blocks_value,file_name,next_val2),stats_file,indent=4)
        
         
        
        
        

scratch_parser_inst = scratch_parser()
scratch_parser_inst.read_files("files/scratch_parent_4.sb3")

    
