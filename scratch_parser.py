import os
import json
import sys
import collections
from unzip_scratch import unzip_scratch
import tempfile
import random
from io import BytesIO
import zipfile
import io
import zlib
#import ijson

class scratch_parser:

    def __init__(self):
        
        self.blocs_json = None
        self.blocks_values = []
        self.final_list_result = []
        self.all_parents_keys_val = []
        self.scr_pro = ""
        self.named_tempfile  = None
        self.sb3class = unzip_scratch()
        self.ommited_block_keys_parent = {"opcode"}
        self.new_connections = []
        self.all_opcodes = []
        self.parse_error=None
        self.scratch_tree_list = []
        self.scratch_stats = {}
        self.next_val_tree = {}
        self.input_block = {}
        self.fin_val = None
        self.sec_val = None
        self.loaded_block = None
        self.in_val = None
        self.new_parent_tree_met = {}
        self.all_met = []
        self.inpt_2 = []
        self.missed_inp  = []
        self.missed_inp2  = []
        self.next_val_dump = None
        self.child_input_keys = []
        self.flat = []
        self.named_tempfile_pars = None
        self.edge = 0
        self.substack_replacement = {"control_repeat":"BodyBlock","control_forever":"BodyBlock","control_if":"ThenBranch","control_if_else":["ThenBranch","BodyBlock"],"control_repeat_until":"BodyBlock"}


    
    def get_all_targets(self,json_data):
        #loaded_blocks = self.tem_file_spit(json_data)
        if isinstance(json_data,dict) and bool(json_data):  
            #print("loaded blocks ", loaded_blocks)
            
            #print("target count ", val)
            targests_value = json_data["targets"] if "targets" in json_data.keys() else {}
                  
            return targests_value
    
    def load_json_in_python(self,json_val):
        return json.loads(json_val)
        
    def list_to_dict(self,lst):
        if not isinstance(lst, list):
            return lst
        elif len(lst) == 2 and isinstance(lst[0], str):
           
            return {lst[0]: self.list_to_dict(lst[1])}
        else:
            
            return [self.list_to_dict(sublist) for sublist in lst]
    
    def implement_directed_graph(self,graph,current):
        if current not in graph:
            return None
        
        
        stack_store = [current]
        while len(stack_store) > 0:
            curr = stack_store.pop()
            self.new_connections.append(curr)
       
            if isinstance(curr,str) or isinstance(curr,int) or isinstance(curr,float) or isinstance(curr,bool):
                if isinstance(graph,dict) and curr in graph.keys() and isinstance(graph[curr],list):
                
                    for neigbour in graph[curr]:
                        stack_store.append({current:neigbour})
        self.new_connections.remove(current)
        
        return self.new_connections
    
    def get_connections(self,node,visited,blocks):
       

        tree_parsed = self.create_next_values2_disp(blocks)
        
        if len(tree_parsed[1:]) == 0:
            set_val = list(visited)
            return set_val
        else:
            for each_child in tree_parsed:
                if isinstance(each_child,str) or isinstance(each_child,int) or isinstance(each_child,bool) or isinstance(each_child,float):
                    visited.add([each_child])
                self.get_connections(each_child,node+visited,blocks)

    def get_all_blocks_vals(self,blocks_values):
        all_blocks = {}
        i= 0
        val_data = self.tem_file_spit(blocks_values)
        targ = self.get_all_targets(val_data)
        if isinstance(targ,list) and len(targ) > 0:
            for each_block in targ:
                i+=1
                if isinstance(each_block,dict) and 'blocks' in each_block.keys():
                    all_blocks[f'blocks{i}'] = each_block['blocks']
        return all_blocks
    
    def get_all_blocks_vals_modified(self,blocks_values):
        all_blocks = []
        i= 0
        val_data = self.tem_file_spit(blocks_values)
        targ = self.get_all_targets(blocks_values)
        
        if isinstance(targ,list) and len(targ) > 0:
            for each_block in targ:
                
                if isinstance(each_block,dict) and 'blocks' in each_block.keys():
                    all_blocks.append(each_block['blocks'])
        return all_blocks

        #return {'blocks':each_block['blocks'] for each_block in targ if isinstance(each_block,dict) and 'blocks' in each_block.keys()}
    
    def flatten_tree(self,tree):
        result = []
        stack = [(tree, [])]

        while len(stack) > 0:
            node, path = stack.pop()

            if isinstance(node, list):
                if node:
                    stack.extend((child, path + [index]) for index, child in enumerate(node[::-1]))
            else:
                result.append((node, path))

        return result

    def get_only_blocks(self,block_targ):
        if block_targ == None or block_targ == {}:
            return {}
        all_blocks =  self.get_all_blocks_vals(block_targ)
        return all_blocks['blocks']
    
    def get_all_blockkeys_from_block(self,blocks_val):
        loaded_blocks = self.tem_file_spit(blocks_val)
        all_keys_blocks = [keys for keys in blocks_val.keys() if isinstance(blocks_val,dict) and keys.startswith('blocks')]
        return all_keys_blocks


    def get_any_block_by_id(self,blocks_values,key):
        loaded_blocks = self.tem_file_spit(blocks_values)
        val = blocks_values.keys() if isinstance(blocks_values,dict) and bool(blocks_values) else []
        if key == None or key == '' or blocks_values == None or blocks_values == {}:
            return {}
        val = [blocks_values[keys][key] for keys in self.get_all_blockkeys_from_block(blocks_values) if key in blocks_values[keys].keys()]
        #print("checking ", val[0])
        return val[0] if len(val) > 0 else {}
    
    
    def get_any_block_by_id_modified(self,blocks_values,key):
        #val = blocks_values.keys()
        if key == None or key == '' or blocks_values == None or blocks_values == []:
            return {}
        val_main = [each_block[key] for each_block in blocks_values if isinstance(blocks_values,list) and len(blocks_values) > 0 and isinstance(each_block,dict) and key in each_block.keys()]
        #val = [blocks_values[keys][key] for keys in self.get_all_blockkeys_from_block(blocks_values) if key in blocks_values[keys].keys()]
        return val_main[0] if len(val_main) > 0 else {}
        
    def check_if_id_is_parent(self,blocks_values,block_id):
        if block_id == None or block_id == '' or blocks_values == None or blocks_values == {}:
            return False
        block = self.get_any_block_by_id(blocks_values,block_id)
        if block == None or block == {}:
            return False
        return 'parent' in block.keys() and block["parent"] == None
          
    def get_parent_complete_opcode(self,blocks_values,block_id):
        if block_id == None or block_id == '' or blocks_values == None or blocks_values == {}:
            return ''
        block = self.get_any_block_by_id(blocks_values,block_id)
        if block == None or block == {}:
            return ''
        inputs = block["inputs"] if "inputs" in block.keys() else {}
        fields = block["fields"] if "fields" in block.keys() else {}
        opcode = block["opcode"] if "opcode" in block.keys() else ''
        if opcode.startswith("event"):
            
            if inputs == {} and fields == {} :
                return opcode
            
            if inputs != {}  and fields != {}: 
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and  isinstance(v[1],str) and len(v[1]) > 0:
                            
                            opcode = f'{opcode}_{k}_{v[0]}_{v[1]}'
                            
                        if isinstance(v[0],str) and len(v[0]) > 0:
                            
                            opcode = f'{opcode}_{k}_{v[0]}' 
                
                        
                for k,v in inputs.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1]}'
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1][1]}'
                return opcode
            elif inputs != {}  and fields == {} :
                
                for k,v in inputs.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1]}'
                            
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1][1]}'
                return opcode

            elif inputs == {}  and fields != {} :
                            
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                            opcode = f'{opcode}_{k}_{v[0]}'
                            
                        elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{k}_{v[0]}_{v[1]}'
                return opcode    
            
        
                    
        else:
            return opcode
    
    def get_complete_fields_inputs(self,blocks_values,block_id):
        opcode = ""
        if block_id == "" or block_id == None or blocks_values == {} or blocks_values == None:
            return opcode
        block = self.get_any_block_by_id(blocks_values,block_id)
        if block == None or block == {}:
            return opcode
        inputs = block["inputs"] if "inputs" in block.keys() else {}
        fields = block["fields"] if "fields" in block.keys() else {}
        main_opcode = block["parent"] if "parent" in block.keys() else ''
        
        if main_opcode == None or main_opcode == "":
            
            if inputs == {} and fields == {} :
                    return opcode
            
            if inputs != {}  and fields != {}: 
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and  isinstance(v[1],str) and len(v[1]) > 0:
                            
                            opcode = f'{k}_{v[0]}_{v[1]}'
                            
                        if isinstance(v[0],str) and len(v[0]) > 0:
                            
                            opcode = f'{k}_{v[0]}' 
                
                        
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1]}'
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1][1]}'
                
                return opcode
            elif inputs != {}  and fields == {} :
                
                for k,v in inputs.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{k}_{v[1]}'
                            
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{k}_{v[1][1]}'
                return opcode

            elif inputs == {}  and fields != {} :
                            
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                            opcode = f'{k}_{v[0]}'
                            
                        elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{k}_{v[0]}_{v[1]}'
            return opcode    
        


    def get_complete_fields_inputs_modified(self,blocks_values,block_id):
        opcode = ""
        if block_id == "" or block_id == None or blocks_values == [] or blocks_values == None:
            return opcode
        block = self.get_any_block_by_id_modified(blocks_values,block_id)
        if block == None or block == {}:
            return opcode
        inputs = block["inputs"] if "inputs" in block.keys() else {}
        fields = block["fields"] if "fields" in block.keys() else {}
        main_opcode = block["parent"] if "parent" in block.keys() else ''
        
        if main_opcode == None or main_opcode == "":
            
            if inputs == {} and fields == {} :
                    return opcode
            
            if inputs != {}  and fields != {}: 
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and  isinstance(v[1],str) and len(v[1]) > 0:
                            
                            opcode = f'{k}_{v[0]}_{v[1]}'
                            
                        if isinstance(v[0],str) and len(v[0]) > 0:
                            
                            opcode = f'{k}_{v[0]}' 
                
                        
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1]}'
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{opcode}_{k}_{v[1][1]}'
                
                return opcode
            elif inputs != {}  and fields == {} :
                
                for k,v in inputs.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{k}_{v[1]}'
                            
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            opcode = f'{k}_{v[1][1]}'
                return opcode

            elif inputs == {}  and fields != {} :
                            
                for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                            opcode = f'{k}_{v[0]}'
                            
                        elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{k}_{v[0]}_{v[1]}'
            return opcode  
                   
    def get_opcode_from_id2(self,blocks_values,block_id):
       if block_id == None or block_id == '' or block_id == None or blocks_values == {} or blocks_values == None:
            return '' 
       val = [blocks_values[keys][block_id]['opcode'] for keys in self.get_all_blockkeys_from_block(blocks_values) if blocks_values[keys][block_id]['opcode'] != None and block_id in blocks_values[keys].keys()]
       return val[0] if len(val) > 0 else ''
       #return blocks_values['blocks'][block_id]['opcode'] if blocks_values['blocks'][block_id]['opcode'] != None else ''

    def get_opcode_from_id2_modified(self,blocks_values,block_id):
       if block_id == None or block_id == '' or block_id == None or blocks_values == [] or blocks_values == None:
            return ''
       val_main = [each_block[block_id]['opcode'] for each_block in blocks_values  if isinstance(blocks_values,list) and len(blocks_values) > 0 and isinstance(each_block,dict) and block_id in each_block.keys()] 
       #val = [blocks_values[keys][block_id]['opcode'] for keys in self.get_all_blockkeys_from_block(blocks_values) if blocks_values[keys][block_id]['opcode'] != None and block_id in blocks_values[keys].keys()]
       return val_main[0]

    def get_opcode_from_id_main(self,block_values,block_id):
        field_block = [block_values[keys][block_id]['fields'] for keys in self.get_all_blockkeys_from_block(block_values) if block_id in block_values[keys].keys()]
        opcode_block = [block_values[keys][block_id]['opcode'] for keys in self.get_all_blockkeys_from_block(block_values) if block_id in block_values[keys].keys()]
        if block_id == None or block_id == '':
            return ''
        
        
        elif field_block[0] == {} or field_block[0] == None:
            return opcode_block[0] if opcode_block[0] != None else ''
        
        if self.check_if_id_is_parent(block_values,block_id):
            return self.get_parent_complete_opcode(block_values,block_id)
        
        elif field_block[0] == {} or field_block[0] == None:
            return opcode_block[0] if opcode_block[0] != None else ''

        else:
            block = self.get_any_block_by_id(block_values,block_id)
            opcode = block["opcode"] if "opcode" in block.keys() else ''
            if block == None or block == {}:
                return ''
            fields = block["fields"] if "fields" in block.keys() else {}
            if fields == {} or fields == None:
                return opcode
            for k,v in fields.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                            opcode = f'{opcode}_{v[0]}'
                        elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and len(v[1]) > 0:
                            opcode = f'{opcode}_{v[0]}_{v[1]}'
            return opcode
    
    def get_opcode_from_id(self,block_values,block_id):
        val = [block_values[keys] for keys in self.get_all_blockkeys_from_block(block_values)]
        val2 = [block_values[keys][block_id] for keys in self.get_all_blockkeys_from_block(block_values) if block_id in block_values[keys].keys()]
        if len(val2) < 1 and block_id == '' or block_id == None or block_values == {} or block_values == None or val == [] or val == None:
            return ''
        
        return val2[0]['opcode'] if len(val2) > 0 and val2[0]['opcode'] != None   else ''
    
    
    def get_opcode_from_id_modified(self,block_values,block_id):
        val_main = [each_block[block_id] for each_block in block_values if isinstance(block_values,list) and len(block_values) > 0 and isinstance(each_block,dict) and block_id in each_block.keys()]
       
        if block_id == '' or block_id == None or block_values == [] or block_values == None or val_main == [] or val_main == None:
            return ''
        
        return val_main[0]['opcode'] if val_main[0]['opcode'] != None or val_main[0]['opcode'] != ''  else ''
                  
    def get_fields_values(self,blocks_values,block_id):
        if block_id == None or block_id == '' or blocks_values == None or blocks_values == {}:
            return ""
        block = self.get_any_block_by_id(blocks_values,block_id)
        if block == None or block == {}:
            return ""
        fields = block["fields"] if "fields" in block.keys() else {}
        if fields != {} or fields != None:
            for k,v in fields.items():
                if isinstance(v,list) and len(v) > 0:
                    if isinstance(v[0],str) and len(v[0]) > 0 and v[1] == None:
                        return f'{k}_{v[0]}' if len(k) > 0 else f'{v[0]}'
                    elif isinstance(v[0],str) and len(v[0]) > 0 and isinstance(v[1],str) and len(v[1]) > 0:
                        return f'{v[0]}{v[1]}'
        else:
            return ""
        
    def get_input_values_parent(self,blocks_values,block_id):
        if block_id == "" or block_id == None or blocks_values == {} or blocks_values == None:
            return ""
        if self.check_if_id_is_parent(blocks_values,block_id):
            return self.get_parent_complete_opcode(blocks_values,block_id)
   
    def return_all_opcodes(self,blocks_values):
        return [self.get_opcode_from_id(blocks_values,k2) for k,v in blocks_values.items() for k2,v2 in v.items() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2)]
    
    def return_all_opcodes_modified(self,blocks_values):
        return [self.get_opcode_from_id_modified(blocks_values,k2) for v in blocks_values for k2,v2 in v.items() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2)]
    

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
    
    def get_all_unique_opcodes_modified(self,blocks_values):
        all_unique_opcodes = []
        if blocks_values == None or blocks_values == []:
            return []
        if isinstance(blocks_values,list) and len(blocks_values) > 0:
            for v in blocks_values:
                if isinstance(v,dict) and bool(v):
                    for k2 in v.keys():
                        opcodes = self.get_opcode_from_id_modified(blocks_values,k2)
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
        
        if isinstance(blocks_values,dict) and bool(blocks_values):
            vall = [blocks_values[keys][id] for keys in self.get_all_blockkeys_from_block(blocks_values) if id in blocks_values[keys].keys()]
            #block = blocks_values['blocks']
            
            
                
            if len(vall) > 0 and isinstance(vall[0],dict) and bool(vall[0]) and 'inputs' in vall[0].keys():
                return vall[0]['inputs']
            
    def read_input_values_by_id_modified(self,blocks_values,id):
        if id == None or id == '' or blocks_values == None or blocks_values == []:
            return {}
        
        if isinstance(blocks_values,list) and len(blocks_values) > 0:
            vall = [each_block[id] for each_block in blocks_values if isinstance(each_block,dict) and id in each_block.keys()]
                
            if len(vall) > 0  and isinstance(vall[0],dict) and bool(vall[0]) and 'inputs' in vall[0].keys():
                return vall[0]['inputs']
            
    def check_dict_depth(self,dict_val,depth=1):
        if not isinstance(dict_val,dict) or not bool(dict_val):
            return depth
        return max(self.check_dict_depth(v,depth+1) for k,v in dict_val.items())
  
    def get_children_key_recursively(self,blocks_values,spec_block):
        if spec_block == None or spec_block == {} or blocks_values == None or blocks_values == {}:
            return []
        
        loaded_blocks = self.tem_file_spit(blocks_values)
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
    
    def get_children_key_recursively_modified(self,blocks_values,spec_block):
        if spec_block == None or spec_block == {} or blocks_values == None or blocks_values == []:
            return []
        
        inp_block = spec_block["inputs"] if isinstance(spec_block,dict)  and "inputs" in spec_block.keys() else {}
        if isinstance(inp_block,dict) and bool(inp_block):
            for k,v in inp_block.items():
                if isinstance(v,list) and len(v) > 0:
                    for each_val in v:
                        if isinstance(each_val,str):
                            if len(each_val) > 0:
                                self.child_input_keys.append(each_val)
                                bloc = self.get_any_block_by_id_modified(blocks_values,each_val)
                                if bloc["inputs"] != None or bloc["inputs"] != {}:
                                    self.get_children_key_recursively_modified(blocks_values,bloc) 
                                else:
                                    break                
        return self.child_input_keys
    
    
    def get_next_child_keys(self,blocks_values,inp_block):
        all_next_keys = []
        loaded_blocks = self.tem_file_spit(blocks_values)
        all_child_keys = self.get_children_key_recursively(blocks_values,inp_block)
        for each_key in all_child_keys:
            block = self.get_any_block_by_id(blocks_values,each_key)
            if isinstance(block,dict) and bool(block) and 'next' in block.keys():
                all_next_keys.append(block['next'])
        return all_next_keys
    
    def get_next_child_keys_modified(self,blocks_values,inp_block):
        all_next_keys = []
        all_child_keys = self.get_children_key_recursively_modified(blocks_values,inp_block)
        for each_key in all_child_keys:
            block = self.get_any_block_by_id_modified(blocks_values,each_key)
            if isinstance(block,dict) and bool(block) and 'next' in block.keys():
                all_next_keys.append(block['next'])
        return all_next_keys

    def get_all_parent_keys(self,blocks_values):
        all_parent_keys = []
        if blocks_values == None or blocks_values == {}:
            return []
        loaded_blocks = self.tem_file_spit(blocks_values)
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2,v2 in v.items():
                        if isinstance(v2,dict) and bool(v2) and 'parent' in v2.keys() and v2['parent'] == None:
                            all_parent_keys.append(k2)
        return all_parent_keys
    
    def get_all_parent_keys_modified(self,blocks_values):
        self.all_parents_keys_val = []
        if blocks_values == None or blocks_values == []:
            return []
        loaded_blocks = self.tem_file_spit(blocks_values)
        if isinstance(blocks_values,list) and len(blocks_values) > 0:
            for v in blocks_values:
                if isinstance(v,dict) and bool(v):
                    for k2,v2 in v.items():
                        if isinstance(v2,dict) and bool(v2) and 'parent' in v2.keys() and v2['parent'] == None:
                            
        
                            self.all_parents_keys_val.append(k2)
        
        return self.all_parents_keys_val
    
    def compare_parent_keys(self,blocks_values,block_key,parent_key):
        if blocks_values == None or blocks_values == {} or block_key == None or block_key == {} or parent_key == None or parent_key == '':
            return False
        loaded_blocks = self.tem_file_spit(blocks_values)
        if isinstance(block_key,dict) and bool(block_key) and 'parent' in block_key.keys():
            parent_block = self.get_any_block_by_id(blocks_values,block_key['parent'])
            if block_key['parent'] != None and block_key['parent'] == parent_key:
                return True
            
            else:
                next_par = self.compare_parent_keys(blocks_values,parent_block,parent_key)
                return next_par
            
    def compare_parent_keys_modified(self,blocks_values,block_key,parent_key):
        if blocks_values == None or blocks_values == [] or block_key == None or block_key == {} or parent_key == None or parent_key == '':
            return False
        
        if isinstance(block_key,dict) and bool(block_key) and 'parent' in block_key.keys():
            parent_block = self.get_any_block_by_id_modified(blocks_values,block_key['parent'])
            if block_key['parent'] != None and block_key['parent'] == parent_key:
                return True
            
            else:
                next_par = self.compare_parent_keys_modified(blocks_values,parent_block,parent_key)
                return next_par
        

    def break_down(self,blocks_values,parent_key):
        spec = []
        if blocks_values == None or blocks_values == {} or parent_key == None or parent_key == '':
            return []
        loaded_blocks = self.tem_file_spit(blocks_values)
        for k,v in blocks_values.items():
            if isinstance(v,dict) and bool(v):
                for k2,v2 in v.items():
                    if isinstance(v2,dict) and bool(v2):
                        if parent_key in self.get_all_parent_keys(blocks_values) and v2["next"] not in self.get_children_key_recursively(blocks_values,self.get_any_block_by_id(blocks_values,k2)) and v2["next"] not in self.get_next_child_keys(blocks_values,self.get_any_block_by_id(blocks_values,k2)) and self.compare_parent_keys(blocks_values,self.get_any_block_by_id(blocks_values,v2["next"]),parent_key):
                            spec.append(v2["next"])
        return spec
    def json_dump_encoder(self,obj):
        if obj is False:
            return "False"
        elif obj is None:
            return "None"
        elif isinstance(obj, (list, dict)):
            return obj
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            return obj
        else:
            return repr(obj)
    
    def tem_file_spit(self,json_obj):
        dumped_val  = json.dumps(json_obj)
        loaded_val = None

        with tempfile.NamedTemporaryFile(mode="w+",delete=False) as fp:
            tem_file_name  = fp.name
        
        with open(tem_file_name,"w") as tmp_file:
            tmp_file.write(dumped_val)
        
        with open(tem_file_name,"r") as rf:
            read_val = rf.read()

            loaded_val = json.loads(read_val)
            os.remove(tem_file_name)
            return loaded_val



        

    def break_down_modified(self,blocks_values,parent_key):
        spec = []
        if blocks_values == None or blocks_values == [] or parent_key == None or parent_key == '':
            return []
        for v in blocks_values:
            if isinstance(v,dict) and bool(v):
                for k2,v2 in v.items():
                    if isinstance(v2,dict) and bool(v2):
                        
                        if parent_key in self.get_all_parent_keys_modified(blocks_values) and v2["next"] not in self.get_children_key_recursively_modified(blocks_values,self.get_any_block_by_id_modified(blocks_values,k2)) and v2["next"] not in self.get_next_child_keys_modified(blocks_values,self.get_any_block_by_id_modified(blocks_values,k2)) and self.compare_parent_keys_modified(blocks_values,self.get_any_block_by_id_modified(blocks_values,v2["next"]),parent_key):
                            spec.append(v2["next"])
        #print("another ",spec)
        return spec
                        

    def get_all_next_id_test(self,blocks_values):
       all_next_id = {}
       if blocks_values == None or blocks_values == {}:
            return {}   
       loaded_blocks = self.tem_file_spit(blocks_values)
       all_parent_keys = self.get_all_parent_keys(blocks_values)
       for each_value in all_parent_keys:
           val_break  = self.break_down(blocks_values,each_value)
           all_next_id[each_value] = val_break
       return all_next_id
       #return {each_value:self.break_down(loaded_blocks,each_value) for each_value in self.get_all_parent_keys(loaded_blocks)}
    
    def get_all_next_id_test_modified(self,blocks_values):
       all_next_id = {}
       if blocks_values == None or blocks_values == []:
            return {}    

       loaded_blocks = self.tem_file_spit(blocks_values)
       for each_value in self.get_all_parent_keys_modified(blocks_values):
           all_next_id[each_value] = self.break_down_modified(blocks_values,each_value)

       #return all_next_id                                               
       return {each_value:self.break_down_modified(blocks_values,each_value) for each_value in self.get_all_parent_keys_modified(blocks_values)}


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
    
    def get_input_block_by_id_key_disp(self,block_values,bid,key):
        
        if key == None or len(key) < 1  or block_values == {} or block_values == None or bid == None or len(bid) < 1:
            return []
        specific_input_by_id_key = []
        input_block = self.read_input_values_by_id(block_values,bid)
        opcode_par  = self.get_opcode_from_id(block_values,bid)
        
        if isinstance(input_block,dict) and bool(input_block): 
            if opcode_par in self.substack_replacement.keys() :
                if key in input_block.keys():
                    value_block =  input_block[key]
                    if isinstance(value_block,list) and len(value_block) > 0:
                        for each_val in value_block:
                            if opcode_par != "control_if_else":
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id(block_values,each_val)
                                    specific_input_by_id_key = [self.substack_replacement[opcode_par]  if isinstance(key,str) and key.startswith("SUBS")  else key,[opcode]]
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    specific_input_by_id_key = [self.substack_replacement[opcode_par] if isinstance(key,str) and key.startswith("SUBS")  else key,[each_val[1]]]
                            else:
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id(block_values,each_val)
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][0],[opcode]]
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][1],[opcode]]
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][0],[each_val[1]]]
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][1],[each_val[1]]]
                                
            else:
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
    
    def get_input_block_by_id_key_disp_modified(self,block_values,bid,key):
        
        if key == None or len(key) < 1  or block_values == [] or block_values == None or bid == None or len(bid) < 1:
            return []
        specific_input_by_id_key = []
        input_block = self.read_input_values_by_id_modified(block_values,bid)
        opcode_par  = self.get_opcode_from_id_modified(block_values,bid)
        
        if isinstance(input_block,dict) and bool(input_block): 
            if opcode_par in self.substack_replacement.keys() :
                if key in input_block.keys():
                    value_block =  input_block[key]
                    if isinstance(value_block,list) and len(value_block) > 0:
                        for each_val in value_block:
                            if opcode_par != "control_if_else":
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id_modified(block_values,each_val)
                                    specific_input_by_id_key = [self.substack_replacement[opcode_par]  if isinstance(key,str) and key.startswith("SUBS")  else key,[opcode]]
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    specific_input_by_id_key = [self.substack_replacement[opcode_par] if isinstance(key,str) and key.startswith("SUBS")  else key,[each_val[1]]]
                            else:
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id_modified(block_values,each_val)
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][0],[opcode]]
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][1],[opcode]]
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][0],[each_val[1]]]
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key = [self.substack_replacement[opcode_par][1],[each_val[1]]]
                                
            else:
                if key in input_block.keys():
                    value_block =  input_block[key]
                    if isinstance(value_block,list) and len(value_block) > 0:
                        for each_val in value_block:
                            if isinstance(each_val,str) and len(each_val) > 0:
                                opcode = self.get_opcode_from_id_modified(block_values,each_val)
                                specific_input_by_id_key = [key,[opcode]]
                            elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                specific_input_by_id_key = [key,[each_val[1]]]

        return specific_input_by_id_key

    def get_input_block_by_id_key_disp2(self,block_values,bid,key):
        
        if key == None or len(key) < 1  or block_values == {} or block_values == None or bid == None or len(bid) < 1:
            return {}
        specific_input_by_id_key_dict = {}
        input_block = self.read_input_values_by_id(block_values,bid)
        opcode_par  = self.get_opcode_from_id(block_values,bid)
        
        if isinstance(input_block,dict) and bool(input_block): 
            if opcode_par in self.substack_replacement.keys() :
                if key in input_block.keys():
                    value_block =  input_block[key]
                    if isinstance(value_block,list) and len(value_block) > 0:
                        for each_val in value_block:
                            if opcode_par != "control_if_else":
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id(block_values,each_val)
                                    specific_input_by_id_key_dict = {self.substack_replacement[opcode_par]  if isinstance(key,str) and key.startswith("SUBS")  else key:opcode}
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    specific_input_by_id_key_dict = {self.substack_replacement[opcode_par] if isinstance(key,str) and key.startswith("SUBS")  else key:each_val[1]}
                            else:
                                if isinstance(each_val,str) and len(each_val) > 0:
                                    opcode = self.get_opcode_from_id(block_values,each_val)
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key_dict = {self.substack_replacement[opcode_par][0]:opcode}
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key_dict = {self.substack_replacement[opcode_par][1]:opcode}
                                elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                    if isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK"):
                                        specific_input_by_id_key_dict = {self.substack_replacement[opcode_par][0]:each_val[1]}
                                    elif isinstance(key,str) and key.startswith("SUBS") and key.endswith("TACK2"):
                                        specific_input_by_id_key_dict = {self.substack_replacement[opcode_par][1]:each_val[1]}
                                
            else:
                if key in input_block.keys():
                    value_block =  input_block[key]
                    if isinstance(value_block,list) and len(value_block) > 0:
                        for each_val in value_block:
                            if isinstance(each_val,str) and len(each_val) > 0:
                                opcode = self.get_opcode_from_id(block_values,each_val)
                                specific_input_by_id_key_dict = {key:opcode}
                            elif isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0:
                                specific_input_by_id_key_dict = {key:each_val[1]}

        return specific_input_by_id_key_dict
        #return specific_input_by_id_key

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
    
    def get_all_inp_keys(self,blocks_values,input_block,id):
        all_keys_dict = {}
        recur_val = {}
        next_opcode = None
        next_rec =  None
        opcode_par  = self.get_opcode_from_id(blocks_values,id)  
        if input_block == None or input_block == {} or blocks_values == None or blocks_values == {}:
            return {}
        val = ''
        input_block = input_block["inputs"] if input_block["inputs"] != {} or input_block["inputs"] != None else {}
        
        if input_block == {} or input_block == None:
            return {}
        if isinstance(input_block,dict) and bool(input_block):
            
            for k,v in input_block.items():   
                if opcode_par in self.substack_replacement.keys():
                    if opcode_par != "control_if_else":
                        
                        if isinstance(v,list) and len(v) > 0:
                            
                            if isinstance(k,str): 
                                if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                    vals = self.get_input_block_by_id_key_disp2(blocks_values,id,k)
                                    print('see vals', vals)
                                    all_keys_dict.update(vals) 
                                    
                                elif isinstance(v[1],str) and len(v[1]) > 0:
                                    
                                    val = self.substack_replacement[opcode_par] if k.startswith("SUBS") else k
                                    opcode = self.get_opcode_from_id(blocks_values,v[1])  
                                
                                    any_block = self.get_any_block_by_id(blocks_values,v[1]) 
                                     
                                    all_keys_dict.update({val:opcode})
                                    
                                    recur_val = self.get_all_inp_keys(blocks_values,any_block,v[1])
                                    next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else '' 
                                    next_rec  = self.get_all_inp_keys(blocks_values,self.get_any_block_by_id(blocks_values,any_block["next"]),any_block["next"])
                                               
                    else:
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(k,str):
                                if isinstance(v[1],str) and len(v[1]) > 0:
                                    
                                    opcode = self.get_opcode_from_id(blocks_values,v[1]) 
                                    any_block = self.get_any_block_by_id(blocks_values,v[1])   
                                    
                                    recur_val = self.get_all_inp_keys(blocks_values,any_block,v[1])
                                    next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else '' 
                                    next_rec  = self.geht_all_inp_keys(blocks_values,self.get_any_block_by_id(blocks_values,any_block["next"]),any_block["next"])
                                    
                                    if k.endswith("TACK2"):
                                        val = self.substack_replacement[opcode_par][-1]
                                        all_keys_dict.update({val:opcode})
                                    
                                    else:
                                        val = self.substack_replacement[opcode_par][0]
                                        all_keys_dict.update({val:opcode})
                                elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                    
                                    vals = self.get_input_block_by_id_key_disp2(blocks_values,id,k)  
                                    
                                    all_keys_dict.update(vals)
                else:
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(k,str): 
                            if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                vals = self.get_input_block_by_id_key_disp2(blocks_values,id,k)
                                
                                all_keys_dict.update(vals) 
                            elif isinstance(v[1],str) and len(v[1]) > 0:
                                opcode = self.get_opcode_from_id(blocks_values,v[1]) 
                                any_block = self.get_any_block_by_id(blocks_values,v[1])   
                                    
                                recur_val = self.get_all_inp_keys(blocks_values,any_block,v[1])
                                
                                next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else '' 
                                next_rec  = self.get_all_inp_keys(blocks_values,self.get_any_block_by_id(blocks_values,any_block["next"]),any_block["next"]) if any_block["next"] != None else {}
                                
                                all_keys_dict.update({k:opcode})
        
        
        return all_keys_dict               

    def correct_input_block_tree_by_id_disp(self,blocks_values,input_block,ids):
        opcode_par  = self.get_opcode_from_id(blocks_values,ids)    

        corr_block_tree = []
        if input_block == None or input_block == {} or blocks_values == None or blocks_values == {}:
            return []
        if isinstance(input_block,dict) and bool(input_block):
            if opcode_par in self.substack_replacement.keys():
                for k,v in input_block.items():
                    if opcode_par != "control_if_else":
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(v[1] ,str) and len(v[1]) > 0:
                                opcode = self.get_opcode_from_id(blocks_values,v[1])  
                                recur_val = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v[1]),v[1])  
                        
                                any_block = self.get_any_block_by_id(blocks_values,v[1])
                                next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                                next_rec  = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,any_block["next"]),any_block["next"]) 
                                if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,[opcode,[recur_val],next_opcode,[next_rec]]])
                                elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,[opcode,[recur_val]]])
                                elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,opcode])
                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                corr_block_tree.append(self.get_input_block_by_id_key_disp(blocks_values,ids,k))
                    else:
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(v[1] ,str) and len(v[1]) > 0:
                                opcode = self.get_opcode_from_id(blocks_values,v[1])  
                                recur_val = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v[1]),v[1])  
                        
                                any_block = self.get_any_block_by_id(blocks_values,v[1])
                                next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                                next_rec  = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,any_block["next"]),any_block["next"]) 
                                if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],[opcode,[recur_val],next_opcode,[next_rec]]])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],[opcode,[recur_val],next_opcode,[next_rec]]])
                                    else:
                                        corr_block_tree.append([k,[opcode,[recur_val],next_opcode,[next_rec]]])
                                elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],[opcode,[recur_val]]])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],[opcode,[recur_val]]])
                                    else:
                                        corr_block_tree.append([k,[opcode,[recur_val]]])
                                elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],opcode])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],opcode])
                                    else:
                                        corr_block_tree.append([k,opcode])
                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                corr_block_tree.append(self.get_input_block_by_id_key_disp(blocks_values,ids,k))
            else:
                for k,v in input_block.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1] ,str) and len(v[1]) > 0:
                            opcode = self.get_opcode_from_id(blocks_values,v[1])  
                            recur_val = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v[1]),v[1])  
                        
                            any_block = self.get_any_block_by_id(blocks_values,v[1])
                            next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                            next_rec  = self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,any_block["next"]),any_block["next"]) 
                            if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                corr_block_tree.append([k,[opcode,[recur_val],next_opcode,[next_rec]]])
                            elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                corr_block_tree.append([k,[opcode,[recur_val]]])
                            elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                corr_block_tree.append([k,opcode])
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            corr_block_tree.append(self.get_input_block_by_id_key_disp(blocks_values,ids,k))
        return corr_block_tree


    def correct_input_block_tree_by_id_disp_modified(self,blocks_values,input_block,ids):
        opcode_par  = self.get_opcode_from_id_modified(blocks_values,ids)    

        corr_block_tree = []
        if input_block == None or input_block == {} or blocks_values == None or blocks_values == []:
            return []
        if isinstance(input_block,dict) and bool(input_block):
            if opcode_par in self.substack_replacement.keys():
                for k,v in input_block.items():
                    if opcode_par != "control_if_else":
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(v[1] ,str) and len(v[1]) > 0:
                                opcode = self.get_opcode_from_id_modified(blocks_values,v[1])  
                                recur_val = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v[1]),v[1])  
                        
                                any_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                                next_opcode = self.get_opcode_from_id_modified(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                                next_rec  = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,any_block["next"]),any_block["next"]) 
                                if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,[opcode,[recur_val],next_opcode,[next_rec]]])
                                elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,[opcode,[recur_val]]])
                                elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                    corr_block_tree.append([self.substack_replacement[opcode_par]  if isinstance(k,str) and k.startswith("SUBS")  else k,opcode])
                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                corr_block_tree.append(self.get_input_block_by_id_key_disp_modified(blocks_values,ids,k))
                    else:
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(v[1] ,str) and len(v[1]) > 0:
                                opcode = self.get_opcode_from_id_modified(blocks_values,v[1])  
                                recur_val = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v[1]),v[1])  
                        
                                any_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                                next_opcode = self.get_opcode_from_id_modified(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                                next_rec  = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,any_block["next"]),any_block["next"]) 
                                if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],[opcode,[recur_val],next_opcode,[next_rec]]])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],[opcode,[recur_val],next_opcode,[next_rec]]])
                                    else:
                                        corr_block_tree.append([k,[opcode,[recur_val],next_opcode,[next_rec]]])
                                elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],[opcode,[recur_val]]])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],[opcode,[recur_val]]])
                                    else:
                                        corr_block_tree.append([k,[opcode,[recur_val]]])
                                elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                    if isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][0],opcode])
                                    elif isinstance(k,str) and k.startswith("SUBS") and k.endswith("TACK2"):
                                        corr_block_tree.append([self.substack_replacement[opcode_par][1],opcode])
                                    else:
                                        corr_block_tree.append([k,opcode])
                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                corr_block_tree.append(self.get_input_block_by_id_key_disp_modified(blocks_values,ids,k))
            else:
                for k,v in input_block.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1] ,str) and len(v[1]) > 0:
                            opcode = self.get_opcode_from_id_modified(blocks_values,v[1])  
                            recur_val = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v[1]),v[1])  
                        
                            any_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            next_opcode = self.get_opcode_from_id_modified(blocks_values,any_block["next"])  if any_block["next"] != None else {} 
                            next_rec  = self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,any_block["next"]),any_block["next"]) 
                            if any_block["next"] != None and next_rec != [] and len(next_rec) > 0 :
                                corr_block_tree.append([k,[opcode,[recur_val],next_opcode,[next_rec]]])
                            elif any_block["next"] ==  None and next_rec != None or next_rec != [] and len(next_rec) > 0:
                                corr_block_tree.append([k,[opcode,[recur_val]]])
                            elif any_block["next"] == None and next_rec == [] or next_rec == None:
                                corr_block_tree.append([k,opcode])
                        elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            corr_block_tree.append(self.get_input_block_by_id_key_disp_modified(blocks_values,ids,k))
        return corr_block_tree

    def create_next_values2(self,blocks_values,file_name):  
        tr = [] 
        final_tree = []
        
       
        all_val = self.get_all_next_id_test(blocks_values)     
        if all_val == None or all_val == {}:
            return []
        if isinstance(all_val,dict) and bool(all_val):
            for ks,vs in all_val.items():
                if isinstance(vs,list) and len(vs) > 0:
                    if isinstance(ks,str) and ks.startswith("event") or ks.startswith("control"):
                        val =  [[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                        tr.append([ks,val])
                    else:
                        all_par_keys = self.get_all_parent_keys(blocks_values)
                        for each_par in all_par_keys:
                            if self.get_opcode_from_id2(blocks_values, each_par) == ks:
                                blocks = self.get_any_block_by_id(blocks_values,each_par)
                                val = [[self.iterate_procedure_input(blocks_values,blocks),[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)]] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                                tr.append([ks,val])                        
        final_tree = [file_name,tr]
        return final_tree
    
    def create_next_values2_disp(self,blocks_values,file_name):  
        tr = [] 
        final_tree = []
        loaded_dump = self.tem_file_spit(blocks_values)
        
        all_val = self.get_all_next_id_test(blocks_values)  
        
        print("all next id ", all_val)
        if all_val == None or all_val == {}:
            return []
        if isinstance(all_val,dict) and bool(all_val):
            for ks,vs in all_val.items():
                
                if isinstance(vs,list) and len(vs) > 0:
                    if isinstance(self.get_opcode_from_id(blocks_values,ks),str) and self.get_opcode_from_id(blocks_values,ks).startswith("event") or self.get_opcode_from_id(blocks_values,ks).startswith("control"):
                        
                        val =  [[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)] if self.get_complete_fields_inputs(blocks_values,v2) == '' or self.get_complete_fields_inputs(blocks_values,v2) == None else [self.get_opcode_from_id(blocks_values,v2),[self.get_complete_fields_inputs(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)]] for v2 in vs ]
                        print("see value ", val)
                        tr.append([self.get_opcode_from_id(blocks_values,ks),val] if self.get_complete_fields_inputs(blocks_values,ks) == "" or self.get_complete_fields_inputs(blocks_values,ks) == None else [self.get_opcode_from_id(blocks_values,ks),[self.get_complete_fields_inputs(blocks_values,ks),val]])
                        print(" tree ", tr)
                    else:
                        if self.get_opcode_from_id2(blocks_values, ks) == self.get_opcode_from_id(blocks_values,ks):
                            blocks = self.get_any_block_by_id(blocks_values,ks)
                            
                            val = [[self.iterate_procedure_input(blocks_values,blocks),[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)]] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                            print(" see val2 ", val)  
                            tr.append([self.get_opcode_from_id(blocks_values,ks),val])
                            print("tree2 ", tr)  
            #print("see tr",tr)                      
            final_tree = [file_name,tr]
            return final_tree
    
    def create_next_values2_disp_modified(self,blocks_values,file_name):  
        tr = [] 
        final_tree = []
        
        loaded_dump = self.tem_file_spit(blocks_values)
        all_val = self.get_all_next_id_test_modified(blocks_values)     
        
        if all_val == None or all_val == {}:
            return []
        if isinstance(all_val,dict) and bool(all_val):
            for ks,vs in all_val.items():
                
                if isinstance(vs,list) and len(vs) > 0:
                    if isinstance(self.get_opcode_from_id_modified(blocks_values,ks),str) and self.get_opcode_from_id_modified(blocks_values,ks).startswith("event") or self.get_opcode_from_id_modified(blocks_values,ks).startswith("control"):
                        
                        val =  [[self.get_opcode_from_id_modified(blocks_values,v2),self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v2),v2)] if self.get_complete_fields_inputs_modified(blocks_values,v2) == '' or self.get_complete_fields_inputs_modified(blocks_values,v2) == None else [self.get_opcode_from_id_modified(blocks_values,v2),[self.get_complete_fields_inputs_modified(blocks_values,v2),self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v2),v2)]] for v2 in vs ]
                        
                        tr.append([self.get_opcode_from_id_modified(blocks_values,ks),val] if self.get_complete_fields_inputs_modified(blocks_values,ks) == "" or self.get_complete_fields_inputs_modified(blocks_values,ks) == None else [self.get_opcode_from_id_modified(blocks_values,ks),[self.get_complete_fields_inputs_modified(blocks_values,ks),val]])
                    else:
                        if self.get_opcode_from_id2_modified(blocks_values, ks) == self.get_opcode_from_id_modified(blocks_values,ks):
                            blocks = self.get_any_block_by_id_modified(blocks_values,ks)
                            val = [[self.iterate_procedure_input_modified(blocks_values,blocks),[self.get_opcode_from_id_modified(blocks_values,v2),self.correct_input_block_tree_by_id_disp_modified(blocks_values,self.read_input_values_by_id_modified(blocks_values,v2),v2)]] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                                
                            tr.append([self.get_opcode_from_id_modified(blocks_values,ks),val])                        
        final_tree = [file_name,tr]
        return final_tree
    

    def get_first_proc_sec(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != {}:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input(blocks_values,child_block)
                            chil_opc = child_block["opcode"] if "opcode" in child_block.keys() else ''
                            child_list.append(chil_opc)
        return child_list
    
    def get_first_proc_sec_modified(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != []:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input_modified(blocks_values,child_block)
                            chil_opc = child_block["opcode"] if "opcode" in child_block.keys() else ''
                            child_list.append(chil_opc)
        return child_list

    def get_mutation(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != {}:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input(blocks_values,child_block)
                            mutation = child_block["mutation"] if "mutation" in child_block.keys() else {}
                            mut_val = mutation["proccode"] if "proccode" in mutation.keys() else ''
                            
                            mut_val = mut_val.replace(' %s %b ','_') if ' %s %b ' in mut_val else mut_val
                            child_list.append(mut_val)
        return child_list
    
    def get_mutation_modified(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != []:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input_modified(blocks_values,child_block)
                            mutation = child_block["mutation"] if "mutation" in child_block.keys() else {}
                            mut_val = mutation["proccode"] if "proccode" in mutation.keys() else ''
                            
                            mut_val = mut_val.replace(' %s %b ','_') if ' %s %b ' in mut_val else mut_val
                            child_list.append(mut_val)
        return child_list
    
    def get_mutation_input(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != {}:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input(blocks_values,child_block)

                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    child_list.append(opcode_ch) 
            return child_list
    
    def get_mutation_input_modified(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != []:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input_modified(blocks_values,child_block)

                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    child_list.append(opcode_ch) 
            return child_list
        
    def get_mutation_input_val(self,blocks_values,input_block):
        child_list = []
        child_dict = {}    
        if input_block != None or blocks_values != {}:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input(blocks_values,child_block)

                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    fields2 = inner_block["fields"] if "fields" in inner_block.keys() else {}
                                    fields_v = [f'{k2}_{v2[0]}' for k2,v2 in fields2.items() if fields2 != {} or fields2 != None and isinstance(v2,list) and len(v2) > 0 and isinstance(v2[0],str) and len(v2[0]) > 0]
                                    child_dict.update({opcode_ch:fields_v[0] if len(fields_v) > 0 else ''})
                                    #child_list.append(fields_v[0] if len(fields_v) > 0 else '') 
        return child_dict
    
    def get_mutation_input_val_modified(self,blocks_values,input_block):
        child_list = []
        child_dict = {}    
        if input_block != None or blocks_values != []:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input_modified(blocks_values,child_block)

                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    fields2 = inner_block["fields"] if "fields" in inner_block.keys() else {}
                                    fields_v = [f'{k2}_{v2[0]}' for k2,v2 in fields2.items() if fields2 != {} or fields2 != None and isinstance(v2,list) and len(v2) > 0 and isinstance(v2[0],str) and len(v2[0]) > 0]
                                    child_dict.update({opcode_ch:fields_v[0] if len(fields_v) > 0 else ''})
                                    #child_list.append(fields_v[0] if len(fields_v) > 0 else '') 
        return child_dict
                          
    def iterate_procedure_input(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != {}:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input(blocks_values,child_block)
                            chil_opc = child_block["opcode"] if "opcode" in child_block.keys() else ''
                            mutation = child_block["mutation"] if "mutation" in child_block.keys() else {}
                            mut_val = mutation["proccode"] if "proccode" in mutation.keys() else ''
                            
                            mut_val = mut_val.replace(' %s %b ','_') if ' %s %b ' in mut_val else mut_val
                            child_list = [chil_opc,[[mut_val]]]
                            
                            
                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    fields = inner_block["fields"] if "fields" in inner_block.keys() else {}
                                    fields_v = [f'{k2}_{v2[0]}' for k2,v2 in fields.items() if fields != {} or fields != None and isinstance(v2,list) and len(v2) > 0 and isinstance(v2[0],str) and len(v2[0]) > 0]
                                    if isinstance(child_list[-1],list) and len(child_list[-1]) > 0:

                                        child_list[-1].append([opcode_ch,[fields_v[0]] if len(fields_v) > 0 else f'{opcode_ch}']) 
                                    else:
                                        child_list.append([opcode_ch,[fields_v[0]] if len(fields_v) > 0 else f'{opcode_ch}'])
                                        

                                
                return child_list
    
                            
    def iterate_procedure_input_modified(self,blocks_values,input_block):
        child_list = []    
        if input_block != None or blocks_values != []:
            
            inputs = input_block["inputs"] if "inputs" in input_block.keys() else {}
            fields = input_block["fields"] if "fields" in input_block.keys() else {}
            
            
            
            if inputs != {} or inputs != None and fields == {} or fields == None:
                
                for k,v in inputs.items():
                    
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],str) and len(v[1]) == 20:
                            child_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                            if child_block != {} or child_block != None:
                                self.iterate_procedure_input_modified(blocks_values,child_block)
                            chil_opc = child_block["opcode"] if "opcode" in child_block.keys() else ''
                            mutation = child_block["mutation"] if "mutation" in child_block.keys() else {}
                            mut_val = mutation["proccode"] if "proccode" in mutation.keys() else ''
                            
                            mut_val = mut_val.replace(' %s %b ','_') if ' %s %b ' in mut_val else mut_val
                            child_list = [chil_opc,[[mut_val]]]
                            
                            
                            for k,v in child_block["inputs"].items():
                                if isinstance(v,list) and len(v) > 0 and isinstance(v[1],str) and len(v[1]) == 20:
                                    inner_block = self.get_any_block_by_id_modified(blocks_values,v[1])
                                    opcode_ch = inner_block["opcode"] if "opcode" in inner_block.keys() else ''
                                    fields = inner_block["fields"] if "fields" in inner_block.keys() else {}
                                    fields_v = [f'{k2}_{v2[0]}' for k2,v2 in fields.items() if fields != {} or fields != None and isinstance(v2,list) and len(v2) > 0 and isinstance(v2[0],str) and len(v2[0]) > 0]
                                    if isinstance(child_list[-1],list) and len(child_list[-1]) > 0:

                                        child_list[-1].append([opcode_ch,[fields_v[0]] if len(fields_v) > 0 else f'{opcode_ch}']) 
                                    else:
                                        child_list.append([opcode_ch,[fields_v[0]] if len(fields_v) > 0 else f'{opcode_ch}'])
                                        

                                
                return child_list
    def rep_sub(self,block,op):
        if block == None or block == {} or op == None or op == '':
            return ''
        
        else:
            return op

    def count_opcodes(self,blocks_values):
        all_opcodes = self.return_all_opcodes(blocks_values)
        if blocks_values == None or blocks_values == {}:
           return {}
           
        if all_opcodes == None or all_opcodes == []:
               return {}
           
        count_val = collections.Counter(all_opcodes)
        return count_val 
    
    def count_opcodes_modified(self,blocks_values):
        all_opcodes = self.return_all_opcodes_modified(blocks_values)
        if blocks_values == None or blocks_values == {}:
           return {}
           
        if all_opcodes == None or all_opcodes == []:
               return {}
           
        count_val = collections.Counter(all_opcodes)
        return count_val 

    def iterate_tree_for_non_opcodes(self,scratch_tree,blocks_values):
        
        if scratch_tree == [] or scratch_tree == None  or blocks_values == {} or blocks_values == None:
            return []   
        if isinstance(scratch_tree,list) and len(scratch_tree) > 0:
            if len(scratch_tree) == 1 and not isinstance(scratch_tree[0],list) and scratch_tree[0] not in self.get_all_unique_opcodes(blocks_values):  
                self.all_met.append(scratch_tree[0])
                return self.all_met
            else:
                for each_val in scratch_tree:
                    if not isinstance(each_val,list):
                        if each_val not in self.get_all_unique_opcodes(blocks_values):
                            self.all_met.append(each_val)
                        else:
                            continue
                    else:
                        self.iterate_tree_for_non_opcodes(each_val,blocks_values)
                        
            return self.all_met   
        
    def iterate_tree_for_non_opcodes2(self, scratch_tree, blocks_values):
        if not scratch_tree or not blocks_values:
            return []

        flattened_tree = []
        stack = [scratch_tree]

        while stack:
            current_node = stack.pop()
            if isinstance(current_node, list):
                stack.extend(current_node)
            elif current_node not in self.get_all_unique_opcodes(blocks_values):
                flattened_tree.append(current_node)

        self.all_met.extend(flattened_tree)
        return flattened_tree
    
    def iterate_tree_for_non_opcodes2_modified(self, scratch_tree, blocks_values):
        if not scratch_tree or not blocks_values:
            return []

        flattened_tree = []
        stack = [scratch_tree]

        while stack:
            current_node = stack.pop()
            if isinstance(current_node, list):
                stack.extend(current_node)
            elif current_node not in self.get_all_unique_opcodes_modified(blocks_values):
                flattened_tree.append(current_node)

        self.all_met.extend(flattened_tree)
        return flattened_tree
    
    def count_non_opcodes(self,blocks_values,scratch_tree):
        non_opcodes = self.iterate_tree_for_non_opcodes2(scratch_tree,blocks_values)
        if blocks_values == None or blocks_values == {} or scratch_tree == None or scratch_tree == [] or non_opcodes == None or non_opcodes == []:
           return {}
           
        count_val = collections.Counter(non_opcodes)
        return count_val
    
    def count_non_opcodes_modified(self,blocks_values,scratch_tree):
        non_opcodes = self.iterate_tree_for_non_opcodes2_modified(scratch_tree,blocks_values)
        if blocks_values == None or blocks_values == {} or scratch_tree == None or scratch_tree == [] or non_opcodes == None or non_opcodes == []:
           return {}
           
        count_val = collections.Counter(non_opcodes)
        return count_val

    def get_all_keys(self,blocks_values):
        all_keys = [] 
        if blocks_values == None or blocks_values == {}: 
            return []
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2 in v.keys():
                        if k2 != [] or k2 != None or k2 != {}:
                            all_keys.append(k2)
        return all_keys

    def get_spec_key_id_leaf(self,blocks_values,id):
        all_key_val = []
        if blocks_values == {} or blocks_values == None or id == "" or id == None:
            return []
        input_block = self.read_input_values_by_id(blocks_values,id)
        block = self.get_any_block_by_id(blocks_values,id)
        parent = block["parent"] if "parent" in block.keys() else ''
        fields = block["fields"] if "fields" in block.keys() else {}
        if parent != None or parent != '':
            if input_block == {} or input_block == None:
                return []
            if isinstance(input_block,dict) and bool(input_block):
                for k,v in input_block.items():
                    if isinstance(v,list) and len(v) > 0: 
                        if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                            all_key_val.append(k)
                            all_key_val.append(v[1][1])
                        elif isinstance(v[1],str) and len(v[1]) > 0:
                            all_key_val.append(k)
                if fields != {} or fields != None:
                    for k,v in fields.items():
                        if isinstance(v,list) and len(v) > 0:
                            if isinstance(v[0],str) and len(v[0]) > 0:
                                all_key_val.append(f'{k}_{v[0]}' if v[1] == None else f'{v[0]}{v[1]}')
                            
                                
        else:
            all_key_val.append(self.get_complete_fields_inputs(blocks_values,id))
        
        return all_key_val
    
    def get_all(self,blocks,all_keys):
        fin = []
        
        for key in all_keys:
            val = self.get_spec_key_id_leaf(blocks,key)
            if val == [] or val == None:
                continue
            fin.append(val)
        return fin

    def get_node_count(self,blocks,all_leafs):
        if blocks == None or blocks == {} or all_leafs == None or all_leafs == []:
            return []
        if isinstance(all_leafs,list) and len(all_leafs) > 0:
            for each_val in all_leafs:
                if isinstance(each_val,list) and len(each_val) > 0:
                    self.get_node_count(blocks,each_val)
                else:
                    self.flat.append(each_val)
        
        return len(self.flat) + len(self.return_all_opcodes(blocks))
    
    def count_nodes_and_edges(self,tree_list):
        if not isinstance(tree_list,list):
            return 0,0

    def get_total_nodes(self,scratch_tree,block):
        if scratch_tree == []:
            return 0
        if isinstance(scratch_tree,list) and len(scratch_tree) == 1:
            return 1
        total_opcodes  = self.return_all_opcodes(block)
        val = self.iterate_tree_for_non_opcodes2(scratch_tree,block)
        
        print(total_opcodes)
        print(val)
        return len(total_opcodes) + len(val)  

    def get_total_edges(self,scratch_tree):  
        main_edges = 0
        if not isinstance(scratch_tree,list):
            return 0
         
        for each_val in scratch_tree:
            if isinstance(each_val,list) and len(each_val) > 0:
                main_edges += 1
                main_edges += self.get_total_edges(each_val)
        return main_edges

        
    def print_tree_top(self,block_values,filename):
        
        self.edge = 0
        if filename != '' or len(filename) > 0 and block_values != {} or block_values != None:
            
            print(f'{filename}')
            val_sub = None
            for each_value in self.get_all_parent_keys(block_values):
                self.edge += 1
                val =  self.break_down(block_values,each_value)
                print(f'|')
                print(f'+---+{self.get_opcode_from_id(block_values,each_value)}')
                if len(self.get_complete_fields_inputs(block_values,each_value)) > 0 and self.get_opcode_from_id(block_values,each_value).startswith("event") or self.get_opcode_from_id(block_values,each_value).startswith("control"):
                    self.edge += 1
                    print(f'    |')
                    print(f'    +---+{self.get_complete_fields_inputs(block_values,each_value)}') 
                if len(self.get_opcode_from_id(block_values,each_value)) > 0 and self.get_opcode_from_id(block_values,each_value).startswith("procedure"):
                    self.edge += 1
                    proc_input = self.get_any_block_by_id(block_values,each_value)
                    proc_call = self.get_first_proc_sec(block_values,proc_input)
                    mut_cal = self.get_mutation(block_values,proc_input)
                    mut_inp = self.get_mutation_input(block_values,proc_input)
                    mut_inp_val = self.get_mutation_input_val(block_values,proc_input)
                    if isinstance(proc_call,list) and len(proc_call) > 0:
                        for each_val in proc_call:
                            self.edge += 1
                            print(f'    |')
                            print(f'    +---+{each_val}')
                        for each_mut in mut_cal:
                            self.edge += 1
                            print(f'        |')
                            print(f'        +---+{each_mut}')
                        if isinstance(mut_inp_val,dict) and bool(mut_inp_val):
                            for each_val_inp_opcode,each_mut_inp in mut_inp_val.items():
                                self.edge += 2
                                print(f'            |')
                                print(f'            +---+{each_val_inp_opcode}')
                                print(f'                |')
                                print(f'                +---+{each_mut_inp}')

                for v in val:
                    self.edge += 1
                    if len(self.get_opcode_from_id(block_values,v)) > 0: 
                        print(f'        |')
                        print(f'    +---+{self.get_opcode_from_id(block_values,v)}')
                        

                        def iterate_leaf(block,input_block,id):
                           all_dict = {}
                           recur_val = {}
                           next_opcode = ''
                           next_rec = {}
                           another_block = {}
                           opcode_par  = self.get_opcode_from_id(block,id) 
                           
                           if input_block != {} or input_block != None and block != {} or block != None:
                               if isinstance(input_block,dict) and bool(input_block) and "inputs" in input_block.keys():
                                    another_block = input_block["inputs"]

                               if another_block != {} or another_block != None:
                                    if isinstance(another_block,dict) and bool(another_block):
            
                                        for k,v in another_block.items():   
                                            if opcode_par in self.substack_replacement.keys():
                                                if opcode_par != "control_if_else":
                        
                                                    if isinstance(v,list) and len(v) > 0:
                            
                                                        if isinstance(k,str): 
                                                            new_key = self.substack_replacement[opcode_par] if k.startswith("SUBS") else k
                                                            if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                                   
                                                                all_dict.update({new_key:v[1][1]})
                                                        
                                    
                                                            elif isinstance(v[1],str) and len(v[1]) > 0:
                                                                opcode =  self.get_opcode_from_id(block,v[1])
                                                                
                                                                any_block = self.get_any_block_by_id(block,v[1]) 
                                                                recur_val = iterate_leaf(block,any_block,v[1])
                                                                if any_block == None or any_block == {}:
                                                                    all_dict.update({new_key:opcode})
                                                                else:
                                                                    
                                                                    recur_val = iterate_leaf(block,any_block,v[1])
                                                                    next_opcode = self.get_opcode_from_id(block,any_block["next"])
                                                                    next_rec = iterate_leaf(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"])
                                                                    if any_block["next"] != None and recur_val != {} or recur_val != None and next_rec != {} or next_rec != None:
                                                                        all_dict.update({new_key:{opcode:recur_val,next_opcode:next_rec}})
                                                                    elif any_block["next"] == None:
                                                                        if recur_val != {} or recur_val != None and next_rec == {} or next_rec == None:
                                                                            all_dict.update({new_key:{opcode:recur_val}})
                                                                    
                                     
                                                                
                                                                #next_opcode = self.get_opcode_from_id(block,any_block["next"])  if any_block["next"] != None else '' 
                                                                #next_rec  = self.get_all_inp_keys(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"])
                                               
                                                else:
                                                    if isinstance(v,list) and len(v) > 0:
                                                        if isinstance(k,str):

                                                            new_key = ''
                                                            if k.endswith("TACK"):
                                                                new_key = self.substack_replacement[opcode_par][0]
                                                            elif k.endswith("TACK2"):
                                                                new_key = self.substack_replacement[opcode_par][-1]
                                                            else:
                                                                new_key = k
                                                            if isinstance(v[1],str) and len(v[1]) > 0:
                                                                
                                                                opcode = self.get_opcode_from_id(block,v[1]) 
                                                                any_block = self.get_any_block_by_id(block,v[1])  
                                                                if any_block == {} or any_block == None:
                                                                    all_dict.update({new_key:opcode})
                                                                else:
                                                                    recur_val = iterate_leaf(block,any_block,v[1])
                                                                    next_opcode = self.get_opcode_from_id(block,any_block["next"])
                                                                    next_rec = iterate_leaf(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"])
                                                                    if any_block["next"] != None and recur_val != {} or recur_val != None and next_rec != {} or next_rec != None:
                                                                        all_dict.update({new_key:{opcode:recur_val,next_opcode:next_rec}})
                                                                    elif any_block["next"] == None:
                                                                        if recur_val != {} or recur_val != None and next_rec == {} or next_rec == None:
                                                                            all_dict.update({new_key:{opcode:recur_val}})
                                                                
                                        
                                                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                                all_dict.update({new_key:v[1][1]})
                                    
                                                                
                                            else:
                                                if isinstance(v,list) and len(v) > 0:
                                                    
                                                    if isinstance(k,str): 
                                                        if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                            all_dict.update({k:v[1][1]})
                                
                                                        elif isinstance(v[1],str) and len(v[1]) > 0:
                                                            opcode = self.get_opcode_from_id(block,v[1]) 
                                                            any_block = self.get_any_block_by_id(block,v[1])
                                                            if any_block == {} or any_block == None:   
                                                                all_dict.update({k:opcode})
                                                            else:
                                                                recur_val = iterate_leaf(block,any_block,v[1])
                                                                next_opcode = self.get_opcode_from_id(block,any_block["next"])  if any_block["next"] != None else '' 
                                                                next_rec  = iterate_leaf(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"]) 
                                                                if any_block["next"] != None and recur_val != {} or recur_val != None and len(next_opcode) > 0 and next_rec != {}:
                                                                    all_dict.update({k:{opcode:recur_val,next_opcode:next_rec}})
                                                                elif any_block["next"] == None and next_rec == {} or next_rec == None:
                                                                    all_dict.update({k:{opcode:recur_val}})
                                                            
                                
                                                            #next_opcode = self.get_opcode_from_id(block,any_block["next"])  if any_block["next"] != None else '' 
                                                            #next_rec  = self.get_all_inp_keys(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"]) if any_block["next"] != None else {}
                                        
                                          
                                        return all_dict

                               else:
                                   return {} 
                        vals = iterate_leaf(block_values,self.get_any_block_by_id(block_values,v),v)
                        def flatten(vals):
                            if isinstance(vals,dict) and bool(vals):
                                for keys_inner,vals_inner in vals.items():
                                    if isinstance(vals_inner,dict) and bool(vals_inner):
                                        flatten(vals_inner)
                                    else:
                                        if keys_inner != None and vals_inner != None:
                                            if len(keys_inner) > 0 and len(vals_inner) > 0:
                                                self.edge += 2
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif len(keys_inner) < 1 and len(vals_inner) > 0:
                                                self.edge += 1
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif len(keys_inner) > 0 and len(vals_inner) < 1:
                                                self.edge += 1
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                        else:
                                            if keys_inner == None and vals_inner == None:
                                                self.edge += 2
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif keys_inner == None and vals_inner != None:
                                                self.edge += 1
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif  keys_inner != None and vals_inner == None:
                                                self.edge += 1
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                            
                                        
                        flatten(vals)
            return self.edge
    
    def print_tree_top_modified(self,block_values,filename):
        
        self.edge = 0
        if filename != '' or len(filename) > 0 and block_values != {} or block_values != None:
            
            print(f'{filename}')
            val_sub = None
            for each_value in self.get_all_parent_keys_modified(block_values):
                self.edge += 1
                val =  self.break_down_modified(block_values,each_value)
                print(f'|')
                print(f'+---+{self.get_opcode_from_id_modified(block_values,each_value)}')
                if len(self.get_complete_fields_inputs_modified(block_values,each_value)) > 0 and self.get_opcode_from_id_modified(block_values,each_value).startswith("event") or self.get_opcode_from_id_modified(block_values,each_value).startswith("control"):
                    self.edge += 1
                    print(f'    |')
                    print(f'    +---+{self.get_complete_fields_inputs_modified(block_values,each_value)}') 
                if len(self.get_opcode_from_id_modified(block_values,each_value)) > 0 and self.get_opcode_from_id_modified(block_values,each_value).startswith("procedure"):
                    self.edge += 1
                    proc_input = self.get_any_block_by_id_modified(block_values,each_value)
                    proc_call = self.get_first_proc_sec_modified(block_values,proc_input)
                    mut_cal = self.get_mutation_modified(block_values,proc_input)
                    mut_inp = self.get_mutation_input_modified(block_values,proc_input)
                    mut_inp_val = self.get_mutation_input_val_modified(block_values,proc_input)
                    if isinstance(proc_call,list) and len(proc_call) > 0:
                        for each_val in proc_call:
                            self.edge += 1
                            print(f'    |')
                            print(f'    +---+{each_val}')
                        for each_mut in mut_cal:
                            self.edge += 1
                            print(f'        |')
                            print(f'        +---+{each_mut}')
                        if isinstance(mut_inp_val,dict) and bool(mut_inp_val):
                            for each_val_inp_opcode,each_mut_inp in mut_inp_val.items():
                                self.edge += 2
                                print(f'            |')
                                print(f'            +---+{each_val_inp_opcode}')
                                print(f'                |')
                                print(f'                +---+{each_mut_inp}')

                for v in val:
                    self.edge += 1
                    if len(self.get_opcode_from_id_modified(block_values,v)) > 0: 
                        print(f'        |')
                        print(f'    +---+{self.get_opcode_from_id_modified(block_values,v)}')
                        

                        def iterate_leaf(block,input_block,id):
                           all_dict = {}
                           recur_val = {}
                           next_opcode = ''
                           next_rec = {}
                           another_block = {}
                           opcode_par  = self.get_opcode_from_id_modified(block,id) 
                           
                           if input_block != {} or input_block != None and block != {} or block != None:
                               if isinstance(input_block,dict) and bool(input_block) and "inputs" in input_block.keys():
                                    another_block = input_block["inputs"]

                               if another_block != {} or another_block != None:
                                    if isinstance(another_block,dict) and bool(another_block):
            
                                        for k,v in another_block.items():   
                                            if opcode_par in self.substack_replacement.keys():
                                                if opcode_par != "control_if_else":
                        
                                                    if isinstance(v,list) and len(v) > 0:
                            
                                                        if isinstance(k,str): 
                                                            new_key = self.substack_replacement[opcode_par] if k.startswith("SUBS") else k
                                                            if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                                   
                                                                all_dict.update({new_key:v[1][1]})
                                                        
                                    
                                                            elif isinstance(v[1],str) and len(v[1]) > 0:
                                                                opcode =  self.get_opcode_from_id_modified(block,v[1])
                                                                
                                                                any_block = self.get_any_block_by_id_modified(block,v[1]) 
                                                                recur_val = iterate_leaf(block,any_block,v[1])
                                                                if any_block == None or any_block == {}:
                                                                    all_dict.update({new_key:opcode})
                                                                else:
                                                                    
                                                                    recur_val = iterate_leaf(block,any_block,v[1])
                                                                    next_opcode = self.get_opcode_from_id_modified(block,any_block["next"])
                                                                    next_rec = iterate_leaf(block,self.get_any_block_by_id_modified(block,any_block["next"]),any_block["next"])
                                                                    if any_block["next"] != None and recur_val != {} or recur_val != None and next_rec != {} or next_rec != None:
                                                                        all_dict.update({new_key:{opcode:recur_val,next_opcode:next_rec}})
                                                                    elif any_block["next"] == None:
                                                                        if recur_val != {} or recur_val != None and next_rec == {} or next_rec == None:
                                                                            all_dict.update({new_key:{opcode:recur_val}})
                                                                    
                                     
                                                                
                                                                #next_opcode = self.get_opcode_from_id(block,any_block["next"])  if any_block["next"] != None else '' 
                                                                #next_rec  = self.get_all_inp_keys(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"])
                                               
                                                else:
                                                    if isinstance(v,list) and len(v) > 0:
                                                        if isinstance(k,str):

                                                            new_key = ''
                                                            if k.endswith("TACK"):
                                                                new_key = self.substack_replacement[opcode_par][0]
                                                            elif k.endswith("TACK2"):
                                                                new_key = self.substack_replacement[opcode_par][-1]
                                                            else:
                                                                new_key = k
                                                            if isinstance(v[1],str) and len(v[1]) > 0:
                                                                
                                                                opcode = self.get_opcode_from_id_modified(block,v[1]) 
                                                                any_block = self.get_any_block_by_id_modified(block,v[1])  
                                                                if any_block == {} or any_block == None:
                                                                    all_dict.update({new_key:opcode})
                                                                else:
                                                                    recur_val = iterate_leaf(block,any_block,v[1])
                                                                    next_opcode = self.get_opcode_from_id_modified(block,any_block["next"])
                                                                    next_rec = iterate_leaf(block,self.get_any_block_by_id_modified(block,any_block["next"]),any_block["next"])
                                                                    if any_block["next"] != None and recur_val != {} or recur_val != None and next_rec != {} or next_rec != None:
                                                                        all_dict.update({new_key:{opcode:recur_val,next_opcode:next_rec}})
                                                                    elif any_block["next"] == None:
                                                                        if recur_val != {} or recur_val != None and next_rec == {} or next_rec == None:
                                                                            all_dict.update({new_key:{opcode:recur_val}})
                                                                
                                        
                                                            elif isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                                all_dict.update({new_key:v[1][1]})
                                    
                                                                
                                            else:
                                                if isinstance(v,list) and len(v) > 0:
                                                    
                                                    if isinstance(k,str): 
                                                        if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and len(v[1][1]) > 0:
                                                            all_dict.update({k:v[1][1]})
                                
                                                        elif isinstance(v[1],str) and len(v[1]) > 0:
                                                            opcode = self.get_opcode_from_id_modified(block,v[1]) 
                                                            any_block = self.get_any_block_by_id_modified(block,v[1])
                                                            if any_block == {} or any_block == None:   
                                                                all_dict.update({k:opcode})
                                                            else:
                                                                recur_val = iterate_leaf(block,any_block,v[1])
                                                                next_opcode = self.get_opcode_from_id_modified(block,any_block["next"])  if any_block["next"] != None else '' 
                                                                next_rec  = iterate_leaf(block,self.get_any_block_by_id_modified(block,any_block["next"]),any_block["next"]) 
                                                                if any_block["next"] != None and recur_val != {} or recur_val != None and len(next_opcode) > 0 and next_rec != {}:
                                                                    all_dict.update({k:{opcode:recur_val,next_opcode:next_rec}})
                                                                elif any_block["next"] == None and next_rec == {} or next_rec == None:
                                                                    all_dict.update({k:{opcode:recur_val}})
                                                            
                                
                                                            #next_opcode = self.get_opcode_from_id(block,any_block["next"])  if any_block["next"] != None else '' 
                                                            #next_rec  = self.get_all_inp_keys(block,self.get_any_block_by_id(block,any_block["next"]),any_block["next"]) if any_block["next"] != None else {}
                                        
                                          
                                        return all_dict

                               else:
                                   return {} 
                        vals = iterate_leaf(block_values,self.get_any_block_by_id_modified(block_values,v),v)
                        def flatten(vals):
                            if isinstance(vals,dict) and bool(vals):
                                for keys_inner,vals_inner in vals.items():
                                    if isinstance(vals_inner,dict) and bool(vals_inner):
                                        flatten(vals_inner)
                                    else:
                                        if keys_inner != None and vals_inner != None:
                                            if len(keys_inner) > 0 and len(vals_inner) > 0:
                                                self.edge += 2
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif len(keys_inner) < 1 and len(vals_inner) > 0:
                                                self.edge += 1
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif len(keys_inner) > 0 and len(vals_inner) < 1:
                                                self.edge += 1
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                        else:
                                            if keys_inner == None and vals_inner == None:
                                                self.edge += 2
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif keys_inner == None and vals_inner != None:
                                                self.edge += 1
                                                print(f'                |')
                                                print(f'                +---+{vals_inner}')
                                            elif  keys_inner != None and vals_inner == None:
                                                self.edge += 1
                                                print(f'            |')
                                                print(f'            +---+{keys_inner}')
                                            
                                        
                        flatten(vals)
            return self.edge
    

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
        
        '''
        for mc in opcodes.most_common(5):
            most_common_opcode_key = mc[0]
            most_common_opcode_val = mc[1]
            most_common_opcode_tree[most_common_opcode_key] = most_common_opcode_val
        
        for nmc in non_opcodes.most_common(5):
            most_common_non_opcode_key = nmc[0]
            most_common_non_opcode_val = nmc[1]
            most_common_non_opcode_tree[most_common_non_opcode_key] = most_common_non_opcode_val
        '''

        nodes_val = sum(opcode_tree.values()) + sum(non_opcode_tree.values())
        
        #nodes, edges = self.count_nodes_and_edges(scratch_tree)
        

        gp_tr = self.list_to_dict(scratch_tree)
       
        root = list(gp_tr.keys())
        firs = self.convert_lst_to_nested_list(scratch_tree)[0]
        connec = self.convert_lst_to_nested_list(scratch_tree)
        flt = self.flatten_inner_nested_lists(connec)
        fr = flt[0]
        flt.remove(fr)
        #print('co',connec)
        #connec.remove(firs)
        
        self.scratch_stats = {"number_of_nodes": nodes_val, "number_of_edges" : self.get_accurate_edge_count(scratch_tree),"opcodes_statistics":opcode_tree,"non_opcodes_statistics":non_opcode_tree,"connections":flt,"all_nodes":self.get_all_nodes(blocks_values,scratch_tree,file_name)}
    
        return self.scratch_stats 
    
    def get_accurate_edge_count(self,parsed_tree):
        
        if not isinstance(parsed_tree,list) or isinstance(parsed_tree,list) and len(parsed_tree) < 2:
            return 0
        
        edges = 1 if isinstance(parsed_tree,list) and len(parsed_tree) > 1 else 0
        
        child_nodes = parsed_tree[1]
        if isinstance(child_nodes,list) and len(child_nodes) > 0:
            for each_child in child_nodes:
                edges += 1
                if isinstance(each_child,list)  and len(each_child) > 0:
                    edges += self.get_accurate_edge_count(each_child)    
            return edges
        
    def generate_all_connections(self,tree_parsed,path=None):
        if path is None:
            path = []

        if isinstance(tree_parsed,str):
            return [path + [tree_parsed]]
        
        if not isinstance(tree_parsed,list) or not tree_parsed:
            return [path]
        
        result = []
        for each_node in tree_parsed:
            result.extend(self.generate_all_connections(each_node,path+[tree_parsed[0]] if path else [tree_parsed[0]]))
        return result

    def generate_summary_stats_modified(self,blocks_values,file_name,scratch_tree):
        
        opcodes = self.count_opcodes_modified(blocks_values)
        non_opcodes = self.count_non_opcodes_modified(blocks_values,scratch_tree)
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
        
        '''
        
        
        for mc in opcodes.most_common(5):
            most_common_opcode_key = mc[0]
            most_common_opcode_val = mc[1]
            most_common_opcode_tree[most_common_opcode_key] = most_common_opcode_val
        
        for nmc in non_opcodes.most_common(5):
            most_common_non_opcode_key = nmc[0]
            most_common_non_opcode_val = nmc[1]
            most_common_non_opcode_tree[most_common_non_opcode_key] = most_common_non_opcode_val
        
        '''
        nodes_val = sum(opcode_tree.values()) + sum(non_opcode_tree.values())
        
        #nodes, edges = self.count_nodes_and_edges(scratch_tree)
        

        gp_tr = self.list_to_dict(scratch_tree)
       
        root = list(gp_tr.keys())
        firs = self.convert_lst_to_nested_list(scratch_tree)[0]
        connec = self.convert_lst_to_nested_list(scratch_tree)
        flt = self.flatten_inner_nested_lists(connec)
        fr = flt[0]
        flt.remove(fr)
        #print('co',connec)
        #connec.remove(firs)
        self.scratch_stats = {"number_of_nodes": nodes_val, "number_of_edges" : self.get_accurate_edge_count(scratch_tree),"opcodes_statistics":opcode_tree,"non_opcodes_statistics":non_opcode_tree,"connections":flt,"all_nodes":self.get_all_nodes_modified(blocks_values,scratch_tree,file_name)}
        return self.scratch_stats 

    def convert_to_flat_list(self,tree):
        result = []

        def dfs(node, path):
            nonlocal result
            if isinstance(node, list):
                for index, child in enumerate(node):
                    dfs(child, path + [index])
            else:
                result.append([])
                for index in path:
                    result[-1].append(repr(tree[index]))
                result[-1].append(repr(node))

        dfs(tree, [])
        return result
    
    def convert_to_connections(self,node,tree,dst):
        paths = []
        new_connections = []
        stack = [node]
        
        new_connections.append([node])
        if isinstance(node,str) or isinstance(node,int) or isinstance(node,bool) or isinstance(node,float) and isinstance(tree,dict) and node in tree.keys():
            
            for each_neigbour in tree[node]:
                if each_neigbour == dst:
                    self.convert_to_connections(each_neigbour,tree,dst)
        '''
        while len(stack) > 0:
            current = stack.pop()
            paths.append(current)
            

            if isinstance(tree,dict) and bool(tree) and isinstance(current,str) or isinstance(current,int) or isinstance(current,bool) or isinstance(current,float) and current in tree.keys() and isinstance(tree[current],list):
                for neigbour in tree[current]:
                    stack.append(neigbour)
                    if neigbour == dst:
                        
                        paths.append(neigbour)
                        new_connections.append(paths)
                        print(paths)
            #for each_val in tree:
                #stack.append(each_val)
        '''
        return new_connections
    
    def find_nodes_between(self,source, dest,graph):
        visited = set()
        print(graph)
        def dfs(node):
            nonlocal visited
            visited.add(node)

            print(node,end='')

            if node == dest:
                return
            
            for neigbour in graph[node]:
                print(neigbour)
                if neigbour not in visited:
                    dfs(neigbour)
        if source not in graph or dest not in graph:
            print('source or destination node not found')
            return
        print("nodes between {} and {} :" .format(source,dest),end='')
        dfs(source)
        print()

    def flatten_nested_list(self,lst):
        result = []
        def flatten_helper(sublist,current_path):
            for item in sublist:
                if isinstance(item,list):
                    flatten_helper(item[1],current_path + [item[0]])
                else:
                    result.append(current_path + [item])
        flatten_helper(lst,[])
        return result
    
    def flatten_inner_nested_lists(self,original_list):
        flattened_list = []
        flat_sublist = []

        for sublist in original_list:
            if not isinstance(sublist[0],list) and isinstance(sublist[0],str) or isinstance(sublist[0],int) or isinstance(sublist[0],bool) or sublist[0] is None or sublist[0] == []:
                flat_sublist = [sublist[0]]
            else:
                self.flatten_inner_nested_lists(sublist[0])
            for item in sublist[1:]:
                if isinstance(item, list):
                    self.flatten_inner_nested_lists(item)
                elif isinstance(item,str) or isinstance(item,int) or isinstance(item,bool) or item is None:
                    if item not in flat_sublist:
                        flat_sublist.append(item)
                    if flat_sublist not in flattened_list:
                        flattened_list.append(flat_sublist)

        return flattened_list
    
    def convert_lst_to_nested_list(self,lst, current_path=[]):
        
        if isinstance(lst, list):
            if len(lst) > 0:
                if lst[0] not in current_path:
                    current_path.append(lst[0])

                if current_path not in self.final_list_result:
                    self.final_list_result.append(current_path)

               
                if len(lst) > 1:
                    for item in lst[1]:
                        if isinstance(item,list):
                            self.convert_lst_to_nested_list(item,list(current_path))
                        else:
                            if item not in current_path:
                                current_path.append(item)
                            if current_path not in self.final_list_result:
                                self.final_list_result.append(list(current_path))

                if len(lst) == 1 and not isinstance(lst[0],list) and lst[0] not in current_path:
                    current_path.append(lst[0])
                elif len(lst) == 1 and isinstance(lst[0],list) and lst[0] not in current_path:
                    self.convert_lst_to_nested_list(lst[0],list(current_path))

            else:
                self.final_list_result.append(current_path)
        else:
            if lst not in current_path and not isinstance(lst,list):
                current_path.append(lst)
            if current_path not in self.final_list_result:
                self.final_list_result.append(list(current_path))

        '''
        for item in lst:
            if isinstance(item,list):
                current_path.append(item[0])
                #self.final_list_result.append(list(current_path))
                self.convert_lst_to_nested_list(item[1],current_path)
            else:
                current_path.append(item)
                self.final_list_result.append(list(current_path))
                current_path.pop()
        if current_path:
            current_path.pop()
        '''
      
        return self.final_list_result

    
        
   
    def all_paths_from(self,current,prefix,seen,edge_map):
        if prefix is None:
            prefix = []
        if seen is None:
            seen = {}
        paths = []
        for k in edge_map[current]:
            if k in seen:
                continue
            new_prefix = prefix + [k]
            paths.append(new_prefix)
            seen[k] = 1
            paths = paths + self.all_paths_from(k,new_prefix,seen,edge_map)
        return paths
    def get_all_all_paths(self,nodes,edge_map):
        all_paths = []
        for i in nodes:
            all_paths = all_paths + (self.all_paths_from(i,[],{},edge_map))
        return all_paths
    
    def make_edge_map(self,nodes,edges):
        edge_map = dict([(i,dict()) for i in nodes])
        for a,b in edges:
            edge_map[a][b] = 1
        return edge_map


            
   

    #connections = convert_to_connections(nested_list)
    def get_all_nodes(self,block,tree,file_name):
        all_nodes = []

        opcodes = self.return_all_opcodes(block)
        non_op = self.iterate_tree_for_non_opcodes2(tree,block)
       
        all_nodes.extend(non_op)
        all_nodes.extend(opcodes)
        if file_name in all_nodes:
            all_nodes.remove(file_name)
        return all_nodes
    
    def get_all_nodes_modified(self,block,tree,file_name):
        all_nodes = []

        opcodes = self.return_all_opcodes_modified(block)
        non_op = self.iterate_tree_for_non_opcodes2_modified(tree,block)
       
        all_nodes.extend(non_op)
        all_nodes.extend(opcodes)
        if file_name in all_nodes:
            all_nodes.remove(file_name)
        return all_nodes
    

    def find_paths_final(self,tree, root, destination):
        path = []
        def dfs(node, paths):
            
            
                if node is None:
                    return
                else:
                    print(node)
                    #path.append(node)
                    if node == destination:
                        print(path)
                        #paths.append(list(path))
            
                    for child in tree[1:]:
                        dfs(child,  paths)
                    path.pop()

        paths = []
        dfs(root,  paths)
        return paths

    def check_read(self,parsed_file):
        file_parsed = self.sb3class.unpack_sb3(parsed_file)
        #print(file_parsed)
        print("    ")
        loads_json = json.loads(file_parsed)
        targets = self.get_all_targets(loads_json)
        blocks = self.get_all_blocks_vals(loads_json)
        blocks_modified = self.get_all_blocks_vals_modified(loads_json)
        parent_keys = self.get_all_parent_keys(blocks)
        parent_keys_modified = self.get_all_parent_keys_modified(blocks_modified)
        
        #break_down = self.break_down(blocks,"dXFi$0R|t1}*%]BdX?*O")
        #break_down_modified = self.break_down_modified(blocks_modified,"dXFi$0R|t1}*%]BdX?*O")
        #res = cL~w*cbzioDnF5Tm6+ZC
        #print(break_down)
        #print("  ")
        #print(break_down_modified)
        #all_next_id = self.get_all_next_id_test(blocks)
        #all_next_id_modified = self.get_all_next_id_test_modified(blocks_modified)
        #compl_fields = self.get_complete_fields_inputs(blocks,"V3*%Z`$oo]i50^8sO?3j")
        #compl_fields_modified = self.get_complete_fields_inputs_modified(blocks_modified,"V3*%Z`$oo]i50^8sO?3j")
        #any_block = self.get_any_block_by_id(blocks,"(,427/wAniEtZt3B.v9X")
       # any_block_modified = self.get_any_block_by_id_modified(blocks_modified,"DZ#xrS_mj!+Behkr#E3a")
        #print(f"any block {any_block} any block modified {any_block_modified}")
        #child_key_rec  = self.get_children_key_recursively(blocks,any_block)
        #child_key_rec_mod = self.get_children_key_recursively_modified(blocks_modified,any_block_modified)
        #all_next_child = self.get_next_child_keys(blocks,any_block)
        #all_next_child_modified = self.get_next_child_keys_modified(blocks_modified,any_block_modified)
        #print(f"all next {all_next_child}")

        print("    ")
        #print(f"all next modified {all_next_child_modified}")
        #print(f"any block {any_block} any block modified {any_block_modified}")
        #inp_id = self.get_input_block_by_id_key_disp_modified(blocks_modified,"DZ#xrS_mj!+Behkr#E3a","pt6xf[~B.jl+=z;O~XbE")
        #inp_val = self.read_input_values_by_id(blocks,"]mjxX2[*l[Cy@/NRZS5H")
        #inp_val_modified = self.read_input_values_by_id_modified(blocks_modified,"v*[[rw;8e#{x6iJcr=-_")
        #print(f" input val {inp_val} input val modified {inp_val_modified}")

        #print("complete fields input ", compl_fields)
        print("   ")
        #print("complete fields input modified ", compl_fields_modified)
        print("   ")
        #print(f"break down {break_down} break down modified {break_down_modified}")
        #print(f"all next id {all_next_id} all next id modified {all_next_id_modified}")
        #print(f"parent keys {parent_keys} parent keys modified {parent_keys_modified}")
        #print("input id ", inp_id)
        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        tree = self.create_next_values2_disp_modified(blocks_modified,file_name)
        print("tree ", tree)
        return parent_keys
    
    def read_files(self, parsed_file):
        self.parsed_value = self.sb3class.unpack_sb3(parsed_file)
        #print("unpacked", {self.parsed_value})
        
        self.blocs_json = json.loads(self.parsed_value)
        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
        
        print("all blcoks",all_blocks_value)

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
        
        all_keys = self.get_all_keys(all_blocks_value)
        all = self.get_all(all_blocks_value,all_keys)
        
        #non_opc = self.iterate_tree_for_non_opcodes2(next_val2,all_blocks_value)
        #gp_tr = self.list_to_dict(next_val2)
        #flt = self.flatten_tree(next_val2)
        #print(flt)
        #flt2 = self.convert_to_flat_list(next_val2)
        #print(flt2)
        #root = list(gp_tr.keys())
        #s =set()
        #non_opcode = self.iterate_tree_for_non_opcodes2(next_val2,all_blocks_value)
        #print(non_opcode)
        #opcode = self.return_all_opcodes(all_blocks_value)
        #print(opcode)
        al_no = self.get_all_nodes(all_blocks_value,next_val2,file_name)
        #print(al_no)
        #print(gp_tr)
        #v = gp_tr.values()
        #print(next_val2)
        #all_v = [self.find_nodes_between("event_whenflagclicked","STEPS",each_val) for k,v in gp_tr.items() for each_val in v][0]
        #print(all_v)
        result_list = []
        #val = self.convert_lst_to_nested_list(next_val2)
        #print(val)
        #print(type(gp_tr))
        #res_cont = self.convert_dict_to_list(gp_tr)
        #print(res_cont)
        #print(v)
        #nd_btw = self.find_nodes_between(root[0],"STEPS",gp_tr)
        #print(nd_btw)
        #for v in al_no:
        #print(self.find_paths_final([1,[2,[[4,5]],3,[[6,7]]]],1,6))
        #print(self.convert_to_connections(root[0],gp_tr,"KEY_OPTION_space"))
        f = self.convert_lst_to_nested_list(next_val2)[0]
        #print('for',f)
        v = self.convert_lst_to_nested_list(next_val2)
        #v.remove(f)
        #print(result_list)
        flt = self.flatten_inner_nested_lists(v)
        #print('here',self.implement_directed_graph(gp_tr,root[0]))
        #print('nested',f_flat)
        
        #fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
        
               
        #return fin_val

    def correct_parse(self,parsed_file):
        parsed_value = self.sb3class.unpack_sb3(parsed_file)
        val = json.loads(parsed_value)
    

        if len(parsed_value) > 0:
            self.blocs_json = json.loads(parsed_value)
        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
        #dump_blocks = json.dumps(all_blocks_value,indent=4)
        
        #return all_blocks_value
        #print(all_blocks_value)

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
        fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
        an_dump = json.dumps(fin_val,indent=4)
        return an_dump
    
    def decode_scratch_bytes(self, raw_bytes):   
        
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zp:
                
                json_file = zp.open('project.json')
                
                project_data = json.load(json_file)
                
                self.scr_pro = json.dumps(project_data,indent=4)
                
                return self.scr_pro
                
        except:
            return ""
        #with BytesIO(raw_bytes,'rb') as f:
            #self.scr_pro += self.sb3class.unpack_sb3(f)
          
            #return self.scr_pro
    
    def decode_sb3_withtem(self,raw_bytes):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            self.named_tempfile = fp.name

        with open(self.named_tempfile,"wb") as temp_file:
            temp_file.write(raw_bytes)

        with open(self.named_tempfile,"rb") as read_temp:
            read_data = read_temp.read()

            with zipfile.ZipFile(io.BytesIO(read_data)) as zp:
                json_file = zp.open('project.json')
                
                project_data = json.load(json_file)
                
                self.scr_pro = json.dumps(project_data,indent=4)
                
                os.remove(self.named_tempfile)
                return self.scr_pro
            

    def count_actual_edges(self,tree):
    # Initialize edge count
        edge_count = 0
    
    # Recursive function to traverse the tree and count actual edges
        def traverse(subtree):
            nonlocal edge_count
            if isinstance(subtree, list) and len(subtree) > 1:
                edge_count += 1
            # The first element is the node, the second element is its children
                children = subtree[1]
                if isinstance(children, list):
                    for child in children:
                        if isinstance(child, list):
                            edge_count += 1
                            traverse(child)
    
    # Start traversal from the root
        traverse(tree)
        return edge_count
    '''
    def decode2(self,raw_bytes,file_name):
        with BytesIO(raw_bytes) as f:
            self.scr_proj = self.sb3class.unpack_sb3(f)
            val = json.loads(self.scr_proj)
            all_blocks_value = self.get_all_blocks_vals(val)
            file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
            next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
            fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
            
            
            return fin_val
    '''
    
    def parse_scratch(self,scr_proj,file_name):
         
        
            val = json.loads(scr_proj)
                
            
            all_blocks_value = self.get_all_blocks_vals(val)
                
                
            
            file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
            next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
            with open("parselogs_upd.txt","a") as plg:
                plg.write(f"filename {file_name} val {val} all blocks {all_blocks_value} first tree {next_val2} \n ")
                plg.write(f"\n==============================================================\n")

            fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
            
            #os.remove(self.named_tempfile_pars)
            return fin_val
            
            
        
    def parse_scratch_modified(self,scr_proj,file_name):
        
        val = json.loads(scr_proj)
        all_blocks_value = self.get_all_blocks_vals_modified(val)
            
            
        file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2_disp_modified(all_blocks_value,file_name)
        with open("parselogs2.txt","a") as plg:
                plg.write(f"val {val} all blocks {all_blocks_value} first tree {next_val2} \n ")
                plg.write(f"\n==============================================================\n")
            
        fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats_modified(all_blocks_value,file_name,next_val2)}
        
        return fin_val
        
        

    def parse_scratch_modified2(self,scr_proj,file_name):
        with tempfile.NamedTemporaryFile(mode='w+',delete=False) as fp:
            self.named_tempfile_pars = fp.name
            self.named_tempfile_pars.write(scr_proj)

       

        with open(self.named_tempfile_pars,"r") as read_temp:
            read_data = read_temp.read()

        if len(read_data) > 0:
            val = json.loads(read_data)
            all_blocks_value = self.get_all_blocks_vals_modified(val)
            
            
            file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
            next_val2 = self.create_next_values2_disp_modified(all_blocks_value,file_name)
            
            fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats_modified(all_blocks_value,file_name,next_val2)}
        
            return fin_val
        
        os.remove(self.named_tempfile_pars)
    


        
#def main(filename: str):
    #pars = scratch_parser_inst.read_files(filename)   
    #return pars
        


#if __name__ == "__main__":
    #file_name = sys.argv[1]
    #main(file_name)

scratch_parser_inst = scratch_parser()
#tr = scratch_parser_inst.correct_parse("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/Chicken Clicker remix-4.sb3")
#print(tr)
edg_c = scratch_parser_inst.get_accurate_edge_count([
      "simple_test",
      [
         [
            "event_whenflagclicked",
            [
               [
                  "motion_movesteps",
                  [
                     [
                        "STEPS",
                        [
                           "10"
                        ]
                     ]
                  ]
               ]
            ]
         ]
      ]
   ])
print(edg_c)

'''
con_test = scratch_parser_inst.flatten_inner_nested_lists(scratch_parser_inst.convert_lst_to_nested_list(
    [
    "kalvin_test_case",
    [
        [
            "event_whenflagclicked",
        [
                [
                    "motion_movesteps",
                    [
                        [
                            "STEPS",
                            [
                                "10"
                            ]
                        ]
                    ]
                ],
                [
                    "motion_turnleft",
                    [
                        [
                            "DEGREES",
                            [
                                "15"
                            ]
                        ]
                    ]
                ],
                [
                    "control_if_else",
                    [
                        [
                            "CONDITION",
                            [
                                "sensing_touchingobject",
                                [
                                    [
                                        [
                                            "TOUCHINGOBJECTMENU",
                                            [
                                                "sensing_touchingobjectmenu__mouse_",
                                                [
                                                    []
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ],
                        [
                            "SUBSTACK",
                            [
                                "sensing_askandwait",
                                [
                                    [
                                        [
                                            "QUESTION",
                                            [
                                                "What's your name?"
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ],
                        [
                            "SUBSTACK2",
                            [
                                "sound_playuntildone",
                                [
                                    [
                                        [
                                            "SOUND_MENU",
                                            [
                                                "sound_sounds_menu_Meow",
                                                [
                                                    []
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
        ]
        ]
    ]
]))
'''

#print(con_test)
#val = scratch_parser_inst.check_read("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/complex3.sb3")
#print(val)
#val = scratch_parser.get_any_block_by_id({'blocks1': {'M.hxK,hM)Q@bk=85rp0T': {'opcode': 'event_whenbroadcastreceived', 'next': '%bD(N8zD]GQ.,Zl4bRbd', 'parent': None, 'inputs': {}, 'fields': {'BROADCAST_OPTION': ['Game Over', '`?i+5`LP/(CoksWKG3RL']}, 'shadow': False, 'topLevel': True, 'x': -409, 'y': -870}, '%bD(N8zD]GQ.,Zl4bRbd': {'opcode': 'control_forever', 'next': None, 'parent': 'M.hxK,hM)Q@bk=85rp0T', 'inputs': {'SUBSTACK': [2, 'v?J]t+R9sobr`bKO));b']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'v?J]t+R9sobr`bKO));b': {'opcode': 'sound_playuntildone', 'next': None, 'parent': '%bD(N8zD]GQ.,Zl4bRbd', 'inputs': {'SOUND_MENU': [1, ',:l6=EGVw#+0nf;%h8Iv']}, 'fields': {}, 'shadow': False, 'topLevel': False}, ',:l6=EGVw#+0nf;%h8Iv': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': 'v?J]t+R9sobr`bKO));b', 'inputs': {}, 'fields': {'SOUND_MENU': ['emotional-titanic-flute', None]}, 'shadow': True, 'topLevel': False}}, 'blocks2': {'C23D/+bAbUxTrv:1RzxP': {'opcode': 'event_whenflagclicked', 'next': ']c_DN9ktwJPSD+[-_h0o', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 44, 'y': -1466}, ']c_DN9ktwJPSD+[-_h0o': {'opcode': 'looks_switchbackdropto', 'next': 'V37-D2@$bG!8BV13$!Zz', 'parent': 'C23D/+bAbUxTrv:1RzxP', 'inputs': {'BACKDROP': [1, '8U`qk764%G4gH8n9ieIF']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '8U`qk764%G4gH8n9ieIF': {'opcode': 'looks_backdrops', 'next': None, 'parent': ']c_DN9ktwJPSD+[-_h0o', 'inputs': {}, 'fields': {'BACKDROP': ['backdrop1', None]}, 'shadow': True, 'topLevel': False}, 'V37-D2@$bG!8BV13$!Zz': {'opcode': 'looks_show', 'next': '?+UNT~w3BXEz?f#DZy`;', 'parent': ']c_DN9ktwJPSD+[-_h0o', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '?+UNT~w3BXEz?f#DZy`;': {'opcode': 'control_forever', 'next': None, 'parent': 'V37-D2@$bG!8BV13$!Zz', 'inputs': {'SUBSTACK': [2, 'yd_p9f#IngzGTV~V3=Hh']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'yd_p9f#IngzGTV~V3=Hh': {'opcode': 'motion_pointtowards', 'next': 'UUw%cjUNV?WYn7O{fv~]', 'parent': '?+UNT~w3BXEz?f#DZy`;', 'inputs': {'TOWARDS': [1, 'bK6m@kv)FAnr;l/}HfjN']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'bK6m@kv)FAnr;l/}HfjN': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': 'yd_p9f#IngzGTV~V3=Hh', 'inputs': {}, 'fields': {'TOWARDS': ['_mouse_', None]}, 'shadow': True, 'topLevel': False}, 'UUw%cjUNV?WYn7O{fv~]': {'opcode': 'control_if', 'next': ',J%S|S81q]X[zP9^{`])', 'parent': 'yd_p9f#IngzGTV~V3=Hh', 'inputs': {'CONDITION': [2, 'm%4M,~F5ipSR4G0FP/.('], 'SUBSTACK': [2, '1me78w3pJ{!I-zT-]Q-;']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'm%4M,~F5ipSR4G0FP/.(': {'opcode': 'sensing_keypressed', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'KEY_OPTION': [1, '01)[R?;VDtZ|qsI/CKB@']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '01)[R?;VDtZ|qsI/CKB@': {'opcode': 'sensing_keyoptions', 'next': None, 'parent': 'm%4M,~F5ipSR4G0FP/.(', 'inputs': {}, 'fields': {'KEY_OPTION': ['w', None]}, 'shadow': True, 'topLevel': False}, '1me78w3pJ{!I-zT-]Q-;': {'opcode': 'motion_movesteps', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'STEPS': [1, [4, '5']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, ',J%S|S81q]X[zP9^{`])': {'opcode': 'control_if', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'CONDITION': [2, '!uPYOZkLcwS3Yaq8`Iww'], 'SUBSTACK': [2, 'rpoTX*P$xiR_wApsQK|{']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '!uPYOZkLcwS3Yaq8`Iww': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': ',J%S|S81q]X[zP9^{`])', 'inputs': {'TOUCHINGOBJECTMENU': [1, 'iA%MPkQ5+;]8YbY9amIJ']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'iA%MPkQ5+;]8YbY9amIJ': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': '!uPYOZkLcwS3Yaq8`Iww', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Zombie', None]}, 'shadow': True, 'topLevel': False}, 'rpoTX*P$xiR_wApsQK|{': {'opcode': 'looks_switchbackdropto', 'next': 'ck2)._e?hfzp|$3wa_Wq', 'parent': ',J%S|S81q]X[zP9^{`])', 'inputs': {'BACKDROP': [1, 'GpNUNKj+gW;=uvJqp~Ep']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'GpNUNKj+gW;=uvJqp~Ep': {'opcode': 'looks_backdrops', 'next': None, 'parent': 'rpoTX*P$xiR_wApsQK|{', 'inputs': {}, 'fields': {'BACKDROP': ['Blue Sky 2 ', None]}, 'shadow': True, 'topLevel': False}, 'ck2)._e?hfzp|$3wa_Wq': {'opcode': 'looks_hide', 'next': 'YM7CbUT8Vw?Pb~?^94E7', 'parent': 'rpoTX*P$xiR_wApsQK|{', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'YM7CbUT8Vw?Pb~?^94E7': {'opcode': 'event_broadcast', 'next': ')s5sHEJ?G|eQM0*zKw#M', 'parent': 'ck2)._e?hfzp|$3wa_Wq', 'inputs': {'BROADCAST_INPUT': [1, [11, 'Game Over', '`?i+5`LP/(CoksWKG3RL']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, ')s5sHEJ?G|eQM0*zKw#M': {'opcode': 'control_stop', 'next': None, 'parent': 'YM7CbUT8Vw?Pb~?^94E7', 'inputs': {}, 'fields': {'STOP_OPTION': ['all', None]}, 'shadow': False, 'topLevel': False, 'mutation': {'tagName': 'mutation', 'children': [], 'hasnext': 'false'}}, 'GKIN6^K)vBF@fY1{38m!': {'opcode': 'event_whenflagclicked', 'next': '*xRI^_Oe53h8rVtVTumr', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 47, 'y': -735}, '*xRI^_Oe53h8rVtVTumr': {'opcode': 'data_setvariableto', 'next': 'Hqa}iTkhIAQ+Wb^$YUw:', 'parent': 'GKIN6^K)vBF@fY1{38m!', 'inputs': {'VALUE': [1, [10, '0']]}, 'fields': {'VARIABLE': ['Score', '`jEk@4|i[#Fk?(8x)AV.-my variable']}, 'shadow': False, 'topLevel': False}, 'Hqa}iTkhIAQ+Wb^$YUw:': {'opcode': 'control_forever', 'next': None, 'parent': '*xRI^_Oe53h8rVtVTumr', 'inputs': {'SUBSTACK': [2, 'G}j,hNT[cXqLy*K+}t}A']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'G}j,hNT[cXqLy*K+}t}A': {'opcode': 'control_if', 'next': None, 'parent': 'Hqa}iTkhIAQ+Wb^$YUw:', 'inputs': {'CONDITION': [2, '.Mz3D6{aFT=};E?e0?mm'], 'SUBSTACK': [2, 'WULxY9~=$g+V80IE!Tmh']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '.Mz3D6{aFT=};E?e0?mm': {'opcode': 'sensing_keypressed', 'next': None, 'parent': 'G}j,hNT[cXqLy*K+}t}A', 'inputs': {'KEY_OPTION': [1, 'iV*ZiTLD)^q]SR#fv?(p']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'iV*ZiTLD)^q]SR#fv?(p': {'opcode': 'sensing_keyoptions', 'next': None, 'parent': '.Mz3D6{aFT=};E?e0?mm', 'inputs': {}, 'fields': {'KEY_OPTION': ['space', None]}, 'shadow': True, 'topLevel': False}, 'WULxY9~=$g+V80IE!Tmh': {'opcode': 'sound_play', 'next': ';4[b=j~`[{jr6FIyYjk/', 'parent': 'G}j,hNT[cXqLy*K+}t}A', 'inputs': {'SOUND_MENU': [1, 'xDLKKDz!*:h~65oW6nQE']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'xDLKKDz!*:h~65oW6nQE': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': 'WULxY9~=$g+V80IE!Tmh', 'inputs': {}, 'fields': {'SOUND_MENU': ['rifle-shot-echo', None]}, 'shadow': True, 'topLevel': False}, ';4[b=j~`[{jr6FIyYjk/': {'opcode': 'control_create_clone_of', 'next': '[RIo0bTesL@IFi*[t%#w', 'parent': 'WULxY9~=$g+V80IE!Tmh', 'inputs': {'CLONE_OPTION': [1, 'cQ@q?X+sIed#+Pt+7)73']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'cQ@q?X+sIed#+Pt+7)73': {'opcode': 'control_create_clone_of_menu', 'next': None, 'parent': ';4[b=j~`[{jr6FIyYjk/', 'inputs': {}, 'fields': {'CLONE_OPTION': ['Bullet', None]}, 'shadow': True, 'topLevel': False}, '[RIo0bTesL@IFi*[t%#w': {'opcode': 'control_wait', 'next': None, 'parent': ';4[b=j~`[{jr6FIyYjk/', 'inputs': {'DURATION': [1, [5, '0.2']]}, 'fields': {}, 'shadow': False, 'topLevel': False}}, 'blocks3': {'*XuesH$(rU;CZ,@yboEv': {'opcode': 'event_whenflagclicked', 'next': '|9K7Y:sgA!%9ip7=AMHO', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 19, 'y': 49}, '|9K7Y:sgA!%9ip7=AMHO': {'opcode': 'looks_hide', 'next': 's(#l*BXY37auR-ua_D)@', 'parent': '*XuesH$(rU;CZ,@yboEv', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '1+Iin9A%Ar?bXfp(spxF': {'opcode': 'control_start_as_clone', 'next': '$/1(*,Had=NDQ]:d+`/:', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 22, 'y': 273}, '$/1(*,Had=NDQ]:d+`/:': {'opcode': 'looks_show', 'next': '/NoF~j-iwMIQbiCM!rMN', 'parent': '1+Iin9A%Ar?bXfp(spxF', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/NoF~j-iwMIQbiCM!rMN': {'opcode': 'motion_goto', 'next': '@tg/1x-(z3?td=uh2S3g', 'parent': '$/1(*,Had=NDQ]:d+`/:', 'inputs': {'TO': [1, 'p.?](gie|65$5;t[:GKm']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'p.?](gie|65$5;t[:GKm': {'opcode': 'motion_goto_menu', 'next': None, 'parent': '/NoF~j-iwMIQbiCM!rMN', 'inputs': {}, 'fields': {'TO': ['Survivor', None]}, 'shadow': True, 'topLevel': False}, '@tg/1x-(z3?td=uh2S3g': {'opcode': 'motion_pointtowards', 'next': '0cW.-LDctKr[C9JSiHgM', 'parent': '/NoF~j-iwMIQbiCM!rMN', 'inputs': {'TOWARDS': [1, 'L15F~L/k1~po}h9UW{2J']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'L15F~L/k1~po}h9UW{2J': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': '@tg/1x-(z3?td=uh2S3g', 'inputs': {}, 'fields': {'TOWARDS': ['_mouse_', None]}, 'shadow': True, 'topLevel': False}, '0cW.-LDctKr[C9JSiHgM': {'opcode': 'control_repeat_until', 'next': '/);uyV.(Y.US2rLY$uz4', 'parent': '@tg/1x-(z3?td=uh2S3g', 'inputs': {'CONDITION': [2, '?YsW!rj;UB%Ow3GXAI/Y'], 'SUBSTACK': [2, 'cyXNaFXd$,@E?]@_DhK_']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'wazvHMCR*7Y,YYKHfu/=': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': '?YsW!rj;UB%Ow3GXAI/Y', 'inputs': {'TOUCHINGOBJECTMENU': [1, 'j4gg+qJfw0n$HT@ejg+t']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'j4gg+qJfw0n$HT@ejg+t': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': 'wazvHMCR*7Y,YYKHfu/=', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['_edge_', None]}, 'shadow': True, 'topLevel': False}, 'cyXNaFXd$,@E?]@_DhK_': {'opcode': 'motion_movesteps', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {'STEPS': [1, [4, '20']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/);uyV.(Y.US2rLY$uz4': {'opcode': 'control_delete_this_clone', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '?YsW!rj;UB%Ow3GXAI/Y': {'opcode': 'operator_or', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {'OPERAND1': [2, 'wazvHMCR*7Y,YYKHfu/='], 'OPERAND2': [2, 'vi5.T|yaXk%T4p].8k33']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'vi5.T|yaXk%T4p].8k33': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': '?YsW!rj;UB%Ow3GXAI/Y', 'inputs': {'TOUCHINGOBJECTMENU': [1, '=[)vc6yI/p=p!Q@(bM#s']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '=[)vc6yI/p=p!Q@(bM#s': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': 'vi5.T|yaXk%T4p].8k33', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Zombie', None]}, 'shadow': True, 'topLevel': False}, 's(#l*BXY37auR-ua_D)@': {'opcode': 'looks_gotofrontback', 'next': None, 'parent': '|9K7Y:sgA!%9ip7=AMHO', 'inputs': {}, 'fields': {'FRONT_BACK': ['back', None]}, 'shadow': False, 'topLevel': False}}, 'blocks4': {'bqXyGzon`N83+r2c7wmD': {'opcode': 'event_whenflagclicked', 'next': 'fRO|fy72P6nw~K9SjMfU', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 70, 'y': -275}, 'fRO|fy72P6nw~K9SjMfU': {'opcode': 'looks_hide', 'next': 'oR^QMp!Uq#h4!sz;B#~Q', 'parent': 'bqXyGzon`N83+r2c7wmD', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'oR^QMp!Uq#h4!sz;B#~Q': {'opcode': 'looks_gotofrontback', 'next': ';)9Xe3uJxp-JLonqEQ2|', 'parent': 'fRO|fy72P6nw~K9SjMfU', 'inputs': {}, 'fields': {'FRONT_BACK': ['back', None]}, 'shadow': False, 'topLevel': False}, ';)9Xe3uJxp-JLonqEQ2|': {'opcode': 'control_forever', 'next': None, 'parent': 'oR^QMp!Uq#h4!sz;B#~Q', 'inputs': {'SUBSTACK': [2, 'rSxz;wVoy!dd}^Kdti.S']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'rSxz;wVoy!dd}^Kdti.S': {'opcode': 'control_wait', 'next': 'oOf=GugPe;FdhylZZBAd', 'parent': ';)9Xe3uJxp-JLonqEQ2|', 'inputs': {'DURATION': [1, [5, '2']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'oOf=GugPe;FdhylZZBAd': {'opcode': 'control_create_clone_of', 'next': None, 'parent': 'rSxz;wVoy!dd}^Kdti.S', 'inputs': {'CLONE_OPTION': [1, 'AFa@0;S(~[No/^63VSb?']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'AFa@0;S(~[No/^63VSb?': {'opcode': 'control_create_clone_of_menu', 'next': None, 'parent': 'oOf=GugPe;FdhylZZBAd', 'inputs': {}, 'fields': {'CLONE_OPTION': ['_myself_', None]}, 'shadow': True, 'topLevel': False}, 'llpHw(s60:eEy!SXEUAx': {'opcode': 'control_start_as_clone', 'next': ',ojgTgdz;YO@s8H?RfvT', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 106, 'y': 330}, ',ojgTgdz;YO@s8H?RfvT': {'opcode': 'looks_show', 'next': 'Rk@T$*H^Yuu0vFpXgajB', 'parent': 'llpHw(s60:eEy!SXEUAx', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'Rk@T$*H^Yuu0vFpXgajB': {'opcode': 'motion_gotoxy', 'next': 'QmU3Eldf1Hl{(Uau=OQ,', 'parent': ',ojgTgdz;YO@s8H?RfvT', 'inputs': {'X': [1, [4, '240']], 'Y': [3, '7,-[~JHPw-,HkZcnGpKZ', [4, '-12']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '7,-[~JHPw-,HkZcnGpKZ': {'opcode': 'operator_random', 'next': None, 'parent': 'Rk@T$*H^Yuu0vFpXgajB', 'inputs': {'FROM': [1, [4, '170']], 'TO': [1, [4, '-170']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'QmU3Eldf1Hl{(Uau=OQ,': {'opcode': 'control_forever', 'next': None, 'parent': 'Rk@T$*H^Yuu0vFpXgajB', 'inputs': {'SUBSTACK': [2, 'v%]~us3a974{TSZ/Cb39']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'v%]~us3a974{TSZ/Cb39': {'opcode': 'motion_pointtowards', 'next': 'h1[6F}6P~.%T-mk`dcVb', 'parent': 'QmU3Eldf1Hl{(Uau=OQ,', 'inputs': {'TOWARDS': [1, '1ApyPf#j3F4z,iy@e`o1']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '1ApyPf#j3F4z,iy@e`o1': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': 'v%]~us3a974{TSZ/Cb39', 'inputs': {}, 'fields': {'TOWARDS': ['Survivor', None]}, 'shadow': True, 'topLevel': False}, 'h1[6F}6P~.%T-mk`dcVb': {'opcode': 'motion_movesteps', 'next': 'p:kV()f[,r#*]k:@3pU|', 'parent': 'v%]~us3a974{TSZ/Cb39', 'inputs': {'STEPS': [1, [4, '1']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'p:kV()f[,r#*]k:@3pU|': {'opcode': 'control_if', 'next': None, 'parent': 'h1[6F}6P~.%T-mk`dcVb', 'inputs': {'CONDITION': [2, '%P5`n@dBP!sOtljY1~M1'], 'SUBSTACK': [2, '70U0KKAgTO~a$axeFxVi']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '%P5`n@dBP!sOtljY1~M1': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': 'p:kV()f[,r#*]k:@3pU|', 'inputs': {'TOUCHINGOBJECTMENU': [1, '/k9;;f}WDSaD$PxGca9Q']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/k9;;f}WDSaD$PxGca9Q': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': '%P5`n@dBP!sOtljY1~M1', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Bullet', None]}, 'shadow': True, 'topLevel': False}, '70U0KKAgTO~a$axeFxVi': {'opcode': 'data_changevariableby', 'next': '5IQcd}[th0$)zd-1a8[1', 'parent': 'p:kV()f[,r#*]k:@3pU|', 'inputs': {'VALUE': [1, [4, '1']]}, 'fields': {'VARIABLE': ['Score', '`jEk@4|i[#Fk?(8x)AV.-my variable']}, 'shadow': False, 'topLevel': False}, '5IQcd}[th0$)zd-1a8[1': {'opcode': 'control_wait', 'next': '$1W2j?_Z:vQ}y,#dpV|1', 'parent': '70U0KKAgTO~a$axeFxVi', 'inputs': {'DURATION': [1, [5, '.05']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '$1W2j?_Z:vQ}y,#dpV|1': {'opcode': 'sound_play', 'next': 'D#2k,Hb,kbsAe~Wx[57m', 'parent': '5IQcd}[th0$)zd-1a8[1', 'inputs': {'SOUND_MENU': [1, 'Su:q4(=e704--)v=Xb-6']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'Su:q4(=e704--)v=Xb-6': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': '$1W2j?_Z:vQ}y,#dpV|1', 'inputs': {}, 'fields': {'SOUND_MENU': ['Zombie groan', None]}, 'shadow': True, 'topLevel': False}, 'D#2k,Hb,kbsAe~Wx[57m': {'opcode': 'control_delete_this_clone', 'next': None, 'parent': '$1W2j?_Z:vQ}y,#dpV|1', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}}},"dXFi$0R|t1}*%]BdX?*O")
#print(val)
#print(scratch_parser.get_all_blockkeys_from_block({'blocks1': {'M.hxK,hM)Q@bk=85rp0T': {'opcode': 'event_whenbroadcastreceived', 'next': '%bD(N8zD]GQ.,Zl4bRbd', 'parent': None, 'inputs': {}, 'fields': {'BROADCAST_OPTION': ['Game Over', '`?i+5`LP/(CoksWKG3RL']}, 'shadow': False, 'topLevel': True, 'x': -409, 'y': -870}, '%bD(N8zD]GQ.,Zl4bRbd': {'opcode': 'control_forever', 'next': None, 'parent': 'M.hxK,hM)Q@bk=85rp0T', 'inputs': {'SUBSTACK': [2, 'v?J]t+R9sobr`bKO));b']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'v?J]t+R9sobr`bKO));b': {'opcode': 'sound_playuntildone', 'next': None, 'parent': '%bD(N8zD]GQ.,Zl4bRbd', 'inputs': {'SOUND_MENU': [1, ',:l6=EGVw#+0nf;%h8Iv']}, 'fields': {}, 'shadow': False, 'topLevel': False}, ',:l6=EGVw#+0nf;%h8Iv': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': 'v?J]t+R9sobr`bKO));b', 'inputs': {}, 'fields': {'SOUND_MENU': ['emotional-titanic-flute', None]}, 'shadow': True, 'topLevel': False}}, 'blocks2': {'C23D/+bAbUxTrv:1RzxP': {'opcode': 'event_whenflagclicked', 'next': ']c_DN9ktwJPSD+[-_h0o', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 44, 'y': -1466}, ']c_DN9ktwJPSD+[-_h0o': {'opcode': 'looks_switchbackdropto', 'next': 'V37-D2@$bG!8BV13$!Zz', 'parent': 'C23D/+bAbUxTrv:1RzxP', 'inputs': {'BACKDROP': [1, '8U`qk764%G4gH8n9ieIF']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '8U`qk764%G4gH8n9ieIF': {'opcode': 'looks_backdrops', 'next': None, 'parent': ']c_DN9ktwJPSD+[-_h0o', 'inputs': {}, 'fields': {'BACKDROP': ['backdrop1', None]}, 'shadow': True, 'topLevel': False}, 'V37-D2@$bG!8BV13$!Zz': {'opcode': 'looks_show', 'next': '?+UNT~w3BXEz?f#DZy`;', 'parent': ']c_DN9ktwJPSD+[-_h0o', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '?+UNT~w3BXEz?f#DZy`;': {'opcode': 'control_forever', 'next': None, 'parent': 'V37-D2@$bG!8BV13$!Zz', 'inputs': {'SUBSTACK': [2, 'yd_p9f#IngzGTV~V3=Hh']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'yd_p9f#IngzGTV~V3=Hh': {'opcode': 'motion_pointtowards', 'next': 'UUw%cjUNV?WYn7O{fv~]', 'parent': '?+UNT~w3BXEz?f#DZy`;', 'inputs': {'TOWARDS': [1, 'bK6m@kv)FAnr;l/}HfjN']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'bK6m@kv)FAnr;l/}HfjN': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': 'yd_p9f#IngzGTV~V3=Hh', 'inputs': {}, 'fields': {'TOWARDS': ['_mouse_', None]}, 'shadow': True, 'topLevel': False}, 'UUw%cjUNV?WYn7O{fv~]': {'opcode': 'control_if', 'next': ',J%S|S81q]X[zP9^{`])', 'parent': 'yd_p9f#IngzGTV~V3=Hh', 'inputs': {'CONDITION': [2, 'm%4M,~F5ipSR4G0FP/.('], 'SUBSTACK': [2, '1me78w3pJ{!I-zT-]Q-;']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'm%4M,~F5ipSR4G0FP/.(': {'opcode': 'sensing_keypressed', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'KEY_OPTION': [1, '01)[R?;VDtZ|qsI/CKB@']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '01)[R?;VDtZ|qsI/CKB@': {'opcode': 'sensing_keyoptions', 'next': None, 'parent': 'm%4M,~F5ipSR4G0FP/.(', 'inputs': {}, 'fields': {'KEY_OPTION': ['w', None]}, 'shadow': True, 'topLevel': False}, '1me78w3pJ{!I-zT-]Q-;': {'opcode': 'motion_movesteps', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'STEPS': [1, [4, '5']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, ',J%S|S81q]X[zP9^{`])': {'opcode': 'control_if', 'next': None, 'parent': 'UUw%cjUNV?WYn7O{fv~]', 'inputs': {'CONDITION': [2, '!uPYOZkLcwS3Yaq8`Iww'], 'SUBSTACK': [2, 'rpoTX*P$xiR_wApsQK|{']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '!uPYOZkLcwS3Yaq8`Iww': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': ',J%S|S81q]X[zP9^{`])', 'inputs': {'TOUCHINGOBJECTMENU': [1, 'iA%MPkQ5+;]8YbY9amIJ']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'iA%MPkQ5+;]8YbY9amIJ': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': '!uPYOZkLcwS3Yaq8`Iww', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Zombie', None]}, 'shadow': True, 'topLevel': False}, 'rpoTX*P$xiR_wApsQK|{': {'opcode': 'looks_switchbackdropto', 'next': 'ck2)._e?hfzp|$3wa_Wq', 'parent': ',J%S|S81q]X[zP9^{`])', 'inputs': {'BACKDROP': [1, 'GpNUNKj+gW;=uvJqp~Ep']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'GpNUNKj+gW;=uvJqp~Ep': {'opcode': 'looks_backdrops', 'next': None, 'parent': 'rpoTX*P$xiR_wApsQK|{', 'inputs': {}, 'fields': {'BACKDROP': ['Blue Sky 2 ', None]}, 'shadow': True, 'topLevel': False}, 'ck2)._e?hfzp|$3wa_Wq': {'opcode': 'looks_hide', 'next': 'YM7CbUT8Vw?Pb~?^94E7', 'parent': 'rpoTX*P$xiR_wApsQK|{', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'YM7CbUT8Vw?Pb~?^94E7': {'opcode': 'event_broadcast', 'next': ')s5sHEJ?G|eQM0*zKw#M', 'parent': 'ck2)._e?hfzp|$3wa_Wq', 'inputs': {'BROADCAST_INPUT': [1, [11, 'Game Over', '`?i+5`LP/(CoksWKG3RL']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, ')s5sHEJ?G|eQM0*zKw#M': {'opcode': 'control_stop', 'next': None, 'parent': 'YM7CbUT8Vw?Pb~?^94E7', 'inputs': {}, 'fields': {'STOP_OPTION': ['all', None]}, 'shadow': False, 'topLevel': False, 'mutation': {'tagName': 'mutation', 'children': [], 'hasnext': 'false'}}, 'GKIN6^K)vBF@fY1{38m!': {'opcode': 'event_whenflagclicked', 'next': '*xRI^_Oe53h8rVtVTumr', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 47, 'y': -735}, '*xRI^_Oe53h8rVtVTumr': {'opcode': 'data_setvariableto', 'next': 'Hqa}iTkhIAQ+Wb^$YUw:', 'parent': 'GKIN6^K)vBF@fY1{38m!', 'inputs': {'VALUE': [1, [10, '0']]}, 'fields': {'VARIABLE': ['Score', '`jEk@4|i[#Fk?(8x)AV.-my variable']}, 'shadow': False, 'topLevel': False}, 'Hqa}iTkhIAQ+Wb^$YUw:': {'opcode': 'control_forever', 'next': None, 'parent': '*xRI^_Oe53h8rVtVTumr', 'inputs': {'SUBSTACK': [2, 'G}j,hNT[cXqLy*K+}t}A']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'G}j,hNT[cXqLy*K+}t}A': {'opcode': 'control_if', 'next': None, 'parent': 'Hqa}iTkhIAQ+Wb^$YUw:', 'inputs': {'CONDITION': [2, '.Mz3D6{aFT=};E?e0?mm'], 'SUBSTACK': [2, 'WULxY9~=$g+V80IE!Tmh']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '.Mz3D6{aFT=};E?e0?mm': {'opcode': 'sensing_keypressed', 'next': None, 'parent': 'G}j,hNT[cXqLy*K+}t}A', 'inputs': {'KEY_OPTION': [1, 'iV*ZiTLD)^q]SR#fv?(p']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'iV*ZiTLD)^q]SR#fv?(p': {'opcode': 'sensing_keyoptions', 'next': None, 'parent': '.Mz3D6{aFT=};E?e0?mm', 'inputs': {}, 'fields': {'KEY_OPTION': ['space', None]}, 'shadow': True, 'topLevel': False}, 'WULxY9~=$g+V80IE!Tmh': {'opcode': 'sound_play', 'next': ';4[b=j~`[{jr6FIyYjk/', 'parent': 'G}j,hNT[cXqLy*K+}t}A', 'inputs': {'SOUND_MENU': [1, 'xDLKKDz!*:h~65oW6nQE']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'xDLKKDz!*:h~65oW6nQE': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': 'WULxY9~=$g+V80IE!Tmh', 'inputs': {}, 'fields': {'SOUND_MENU': ['rifle-shot-echo', None]}, 'shadow': True, 'topLevel': False}, ';4[b=j~`[{jr6FIyYjk/': {'opcode': 'control_create_clone_of', 'next': '[RIo0bTesL@IFi*[t%#w', 'parent': 'WULxY9~=$g+V80IE!Tmh', 'inputs': {'CLONE_OPTION': [1, 'cQ@q?X+sIed#+Pt+7)73']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'cQ@q?X+sIed#+Pt+7)73': {'opcode': 'control_create_clone_of_menu', 'next': None, 'parent': ';4[b=j~`[{jr6FIyYjk/', 'inputs': {}, 'fields': {'CLONE_OPTION': ['Bullet', None]}, 'shadow': True, 'topLevel': False}, '[RIo0bTesL@IFi*[t%#w': {'opcode': 'control_wait', 'next': None, 'parent': ';4[b=j~`[{jr6FIyYjk/', 'inputs': {'DURATION': [1, [5, '0.2']]}, 'fields': {}, 'shadow': False, 'topLevel': False}}, 'blocks3': {'*XuesH$(rU;CZ,@yboEv': {'opcode': 'event_whenflagclicked', 'next': '|9K7Y:sgA!%9ip7=AMHO', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 19, 'y': 49}, '|9K7Y:sgA!%9ip7=AMHO': {'opcode': 'looks_hide', 'next': 's(#l*BXY37auR-ua_D)@', 'parent': '*XuesH$(rU;CZ,@yboEv', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '1+Iin9A%Ar?bXfp(spxF': {'opcode': 'control_start_as_clone', 'next': '$/1(*,Had=NDQ]:d+`/:', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 22, 'y': 273}, '$/1(*,Had=NDQ]:d+`/:': {'opcode': 'looks_show', 'next': '/NoF~j-iwMIQbiCM!rMN', 'parent': '1+Iin9A%Ar?bXfp(spxF', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/NoF~j-iwMIQbiCM!rMN': {'opcode': 'motion_goto', 'next': '@tg/1x-(z3?td=uh2S3g', 'parent': '$/1(*,Had=NDQ]:d+`/:', 'inputs': {'TO': [1, 'p.?](gie|65$5;t[:GKm']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'p.?](gie|65$5;t[:GKm': {'opcode': 'motion_goto_menu', 'next': None, 'parent': '/NoF~j-iwMIQbiCM!rMN', 'inputs': {}, 'fields': {'TO': ['Survivor', None]}, 'shadow': True, 'topLevel': False}, '@tg/1x-(z3?td=uh2S3g': {'opcode': 'motion_pointtowards', 'next': '0cW.-LDctKr[C9JSiHgM', 'parent': '/NoF~j-iwMIQbiCM!rMN', 'inputs': {'TOWARDS': [1, 'L15F~L/k1~po}h9UW{2J']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'L15F~L/k1~po}h9UW{2J': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': '@tg/1x-(z3?td=uh2S3g', 'inputs': {}, 'fields': {'TOWARDS': ['_mouse_', None]}, 'shadow': True, 'topLevel': False}, '0cW.-LDctKr[C9JSiHgM': {'opcode': 'control_repeat_until', 'next': '/);uyV.(Y.US2rLY$uz4', 'parent': '@tg/1x-(z3?td=uh2S3g', 'inputs': {'CONDITION': [2, '?YsW!rj;UB%Ow3GXAI/Y'], 'SUBSTACK': [2, 'cyXNaFXd$,@E?]@_DhK_']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'wazvHMCR*7Y,YYKHfu/=': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': '?YsW!rj;UB%Ow3GXAI/Y', 'inputs': {'TOUCHINGOBJECTMENU': [1, 'j4gg+qJfw0n$HT@ejg+t']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'j4gg+qJfw0n$HT@ejg+t': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': 'wazvHMCR*7Y,YYKHfu/=', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['_edge_', None]}, 'shadow': True, 'topLevel': False}, 'cyXNaFXd$,@E?]@_DhK_': {'opcode': 'motion_movesteps', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {'STEPS': [1, [4, '20']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/);uyV.(Y.US2rLY$uz4': {'opcode': 'control_delete_this_clone', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, '?YsW!rj;UB%Ow3GXAI/Y': {'opcode': 'operator_or', 'next': None, 'parent': '0cW.-LDctKr[C9JSiHgM', 'inputs': {'OPERAND1': [2, 'wazvHMCR*7Y,YYKHfu/='], 'OPERAND2': [2, 'vi5.T|yaXk%T4p].8k33']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'vi5.T|yaXk%T4p].8k33': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': '?YsW!rj;UB%Ow3GXAI/Y', 'inputs': {'TOUCHINGOBJECTMENU': [1, '=[)vc6yI/p=p!Q@(bM#s']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '=[)vc6yI/p=p!Q@(bM#s': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': 'vi5.T|yaXk%T4p].8k33', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Zombie', None]}, 'shadow': True, 'topLevel': False}, 's(#l*BXY37auR-ua_D)@': {'opcode': 'looks_gotofrontback', 'next': None, 'parent': '|9K7Y:sgA!%9ip7=AMHO', 'inputs': {}, 'fields': {'FRONT_BACK': ['back', None]}, 'shadow': False, 'topLevel': False}}, 'blocks4': {'bqXyGzon`N83+r2c7wmD': {'opcode': 'event_whenflagclicked', 'next': 'fRO|fy72P6nw~K9SjMfU', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 70, 'y': -275}, 'fRO|fy72P6nw~K9SjMfU': {'opcode': 'looks_hide', 'next': 'oR^QMp!Uq#h4!sz;B#~Q', 'parent': 'bqXyGzon`N83+r2c7wmD', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'oR^QMp!Uq#h4!sz;B#~Q': {'opcode': 'looks_gotofrontback', 'next': ';)9Xe3uJxp-JLonqEQ2|', 'parent': 'fRO|fy72P6nw~K9SjMfU', 'inputs': {}, 'fields': {'FRONT_BACK': ['back', None]}, 'shadow': False, 'topLevel': False}, ';)9Xe3uJxp-JLonqEQ2|': {'opcode': 'control_forever', 'next': None, 'parent': 'oR^QMp!Uq#h4!sz;B#~Q', 'inputs': {'SUBSTACK': [2, 'rSxz;wVoy!dd}^Kdti.S']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'rSxz;wVoy!dd}^Kdti.S': {'opcode': 'control_wait', 'next': 'oOf=GugPe;FdhylZZBAd', 'parent': ';)9Xe3uJxp-JLonqEQ2|', 'inputs': {'DURATION': [1, [5, '2']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'oOf=GugPe;FdhylZZBAd': {'opcode': 'control_create_clone_of', 'next': None, 'parent': 'rSxz;wVoy!dd}^Kdti.S', 'inputs': {'CLONE_OPTION': [1, 'AFa@0;S(~[No/^63VSb?']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'AFa@0;S(~[No/^63VSb?': {'opcode': 'control_create_clone_of_menu', 'next': None, 'parent': 'oOf=GugPe;FdhylZZBAd', 'inputs': {}, 'fields': {'CLONE_OPTION': ['_myself_', None]}, 'shadow': True, 'topLevel': False}, 'llpHw(s60:eEy!SXEUAx': {'opcode': 'control_start_as_clone', 'next': ',ojgTgdz;YO@s8H?RfvT', 'parent': None, 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': True, 'x': 106, 'y': 330}, ',ojgTgdz;YO@s8H?RfvT': {'opcode': 'looks_show', 'next': 'Rk@T$*H^Yuu0vFpXgajB', 'parent': 'llpHw(s60:eEy!SXEUAx', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'Rk@T$*H^Yuu0vFpXgajB': {'opcode': 'motion_gotoxy', 'next': 'QmU3Eldf1Hl{(Uau=OQ,', 'parent': ',ojgTgdz;YO@s8H?RfvT', 'inputs': {'X': [1, [4, '240']], 'Y': [3, '7,-[~JHPw-,HkZcnGpKZ', [4, '-12']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '7,-[~JHPw-,HkZcnGpKZ': {'opcode': 'operator_random', 'next': None, 'parent': 'Rk@T$*H^Yuu0vFpXgajB', 'inputs': {'FROM': [1, [4, '170']], 'TO': [1, [4, '-170']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'QmU3Eldf1Hl{(Uau=OQ,': {'opcode': 'control_forever', 'next': None, 'parent': 'Rk@T$*H^Yuu0vFpXgajB', 'inputs': {'SUBSTACK': [2, 'v%]~us3a974{TSZ/Cb39']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'v%]~us3a974{TSZ/Cb39': {'opcode': 'motion_pointtowards', 'next': 'h1[6F}6P~.%T-mk`dcVb', 'parent': 'QmU3Eldf1Hl{(Uau=OQ,', 'inputs': {'TOWARDS': [1, '1ApyPf#j3F4z,iy@e`o1']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '1ApyPf#j3F4z,iy@e`o1': {'opcode': 'motion_pointtowards_menu', 'next': None, 'parent': 'v%]~us3a974{TSZ/Cb39', 'inputs': {}, 'fields': {'TOWARDS': ['Survivor', None]}, 'shadow': True, 'topLevel': False}, 'h1[6F}6P~.%T-mk`dcVb': {'opcode': 'motion_movesteps', 'next': 'p:kV()f[,r#*]k:@3pU|', 'parent': 'v%]~us3a974{TSZ/Cb39', 'inputs': {'STEPS': [1, [4, '1']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'p:kV()f[,r#*]k:@3pU|': {'opcode': 'control_if', 'next': None, 'parent': 'h1[6F}6P~.%T-mk`dcVb', 'inputs': {'CONDITION': [2, '%P5`n@dBP!sOtljY1~M1'], 'SUBSTACK': [2, '70U0KKAgTO~a$axeFxVi']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '%P5`n@dBP!sOtljY1~M1': {'opcode': 'sensing_touchingobject', 'next': None, 'parent': 'p:kV()f[,r#*]k:@3pU|', 'inputs': {'TOUCHINGOBJECTMENU': [1, '/k9;;f}WDSaD$PxGca9Q']}, 'fields': {}, 'shadow': False, 'topLevel': False}, '/k9;;f}WDSaD$PxGca9Q': {'opcode': 'sensing_touchingobjectmenu', 'next': None, 'parent': '%P5`n@dBP!sOtljY1~M1', 'inputs': {}, 'fields': {'TOUCHINGOBJECTMENU': ['Bullet', None]}, 'shadow': True, 'topLevel': False}, '70U0KKAgTO~a$axeFxVi': {'opcode': 'data_changevariableby', 'next': '5IQcd}[th0$)zd-1a8[1', 'parent': 'p:kV()f[,r#*]k:@3pU|', 'inputs': {'VALUE': [1, [4, '1']]}, 'fields': {'VARIABLE': ['Score', '`jEk@4|i[#Fk?(8x)AV.-my variable']}, 'shadow': False, 'topLevel': False}, '5IQcd}[th0$)zd-1a8[1': {'opcode': 'control_wait', 'next': '$1W2j?_Z:vQ}y,#dpV|1', 'parent': '70U0KKAgTO~a$axeFxVi', 'inputs': {'DURATION': [1, [5, '.05']]}, 'fields': {}, 'shadow': False, 'topLevel': False}, '$1W2j?_Z:vQ}y,#dpV|1': {'opcode': 'sound_play', 'next': 'D#2k,Hb,kbsAe~Wx[57m', 'parent': '5IQcd}[th0$)zd-1a8[1', 'inputs': {'SOUND_MENU': [1, 'Su:q4(=e704--)v=Xb-6']}, 'fields': {}, 'shadow': False, 'topLevel': False}, 'Su:q4(=e704--)v=Xb-6': {'opcode': 'sound_sounds_menu', 'next': None, 'parent': '$1W2j?_Z:vQ}y,#dpV|1', 'inputs': {}, 'fields': {'SOUND_MENU': ['Zombie groan', None]}, 'shadow': True, 'topLevel': False}, 'D#2k,Hb,kbsAe~Wx[57m': {'opcode': 'control_delete_this_clone', 'next': None, 'parent': '$1W2j?_Z:vQ}y,#dpV|1', 'inputs': {}, 'fields': {}, 'shadow': False, 'topLevel': False}}}))
#print(scratch_parser_inst.read_files("files/Chicken Clicker remix-4.sb3"))

    
print(scratch_parser_inst.correct_parse("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/maze_2.sb3"))