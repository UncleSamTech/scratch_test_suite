import os
import json
import sys
import collections
from unzip_scratch import unzip_scratch
import tempfile

from io import BytesIO
import zipfile
import zlib

class scratch_parser:

    def __init__(self):
        
        self.blocs_json = None
        self.blocks_values = []
        self.final_list_result = []
        self.scr_pro = None
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
        self.in_val = None
        self.new_parent_tree_met = {}
        self.all_met = []
        self.inpt_2 = []
        self.missed_inp  = []
        self.missed_inp2  = []
        self.child_input_keys = []
        self.flat = []
        self.edge = 0
        self.substack_replacement = {"control_repeat":"BodyBlock","control_forever":"BodyBlock","control_if":"ThenBranch","control_if_else":["ThenBranch","BodyBlock"],"control_repeat_until":"BodyBlock"}


    
    def get_all_targets(self,json_data):
        if isinstance(json_data,dict) and bool(json_data):          
            return json_data["targets"] if 'targets' in json_data.keys() else {}
        
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
        targ = self.get_all_targets(blocks_values)
        return {'blocks':each_block['blocks'] for each_block in targ if isinstance(each_block,dict) and 'blocks' in each_block.keys()}
    
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
    
    def get_any_block_by_id(self,blocks_values,key):
        if key == None or key == '' or blocks_values == None or blocks_values == {} or blocks_values['blocks'] == None or blocks_values['blocks'] == {} or blocks_values['blocks'][key] == None or blocks_values['blocks'][key] == {}:
            return {}
        return blocks_values['blocks'][key]
        
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
                   
    def get_opcode_from_id2(self,blocks_values,block_id):
       if block_id == None or block_id == '' or block_id == None or blocks_values == {} or blocks_values == None:
            return '' 
       return blocks_values['blocks'][block_id]['opcode'] if blocks_values['blocks'][block_id]['opcode'] != None else ''

    def get_opcode_from_id_main(self,block_values,block_id):
        if block_id == None or block_id == '':
            return ''
        elif block_values['blocks'][block_id]['fields'] == {} or block_values['blocks'][block_id]['fields'] == None:
            return block_values['blocks'][block_id]['opcode'] if block_values['blocks'][block_id]['opcode'] != None else ''
        
        if self.check_if_id_is_parent(block_values,block_id):
            return self.get_parent_complete_opcode(block_values,block_id)
        
        elif block_values['blocks'][block_id]['fields'] == {} or block_values['blocks'][block_id]['fields'] == None:
            return block_values['blocks'][block_id]['opcode'] if block_values['blocks'][block_id]['opcode'] != None else ''

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
        if block_id == '' or block_id == None or block_values == {} or block_values == None or block_values['blocks'] == {} or block_values['blocks'] == None or block_values['blocks'][block_id] == {} or block_values['blocks'][block_id] == None:
            return ''
        
        return block_values['blocks'][block_id]['opcode'] if block_values['blocks'][block_id]['opcode'] != None or block_values['blocks'][block_id]['opcode'] != ''  else ''
                  
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
       return {each_value:self.break_down(blocks_values,each_value) for each_value in self.get_all_parent_keys(blocks_values)}

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
        
        
        all_val = self.get_all_next_id_test(blocks_values)     
        if all_val == None or all_val == {}:
            return []
        if isinstance(all_val,dict) and bool(all_val):
            for ks,vs in all_val.items():
                
                if isinstance(vs,list) and len(vs) > 0:
                    if isinstance(self.get_opcode_from_id(blocks_values,ks),str) and self.get_opcode_from_id(blocks_values,ks).startswith("event") or self.get_opcode_from_id(blocks_values,ks).startswith("control"):
                        
                        val =  [[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)] if self.get_complete_fields_inputs(blocks_values,v2) == '' or self.get_complete_fields_inputs(blocks_values,v2) == None else [self.get_opcode_from_id(blocks_values,v2),[self.get_complete_fields_inputs(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)]] for v2 in vs ]
                        
                        tr.append([self.get_opcode_from_id(blocks_values,ks),val] if self.get_complete_fields_inputs(blocks_values,ks) == "" or self.get_complete_fields_inputs(blocks_values,ks) == None else [self.get_opcode_from_id(blocks_values,ks),[self.get_complete_fields_inputs(blocks_values,ks),val]])
                    else:
                        if self.get_opcode_from_id2(blocks_values, ks) == self.get_opcode_from_id(blocks_values,ks):
                            blocks = self.get_any_block_by_id(blocks_values,ks)
                            val = [[self.iterate_procedure_input(blocks_values,blocks),[self.get_opcode_from_id(blocks_values,v2),self.correct_input_block_tree_by_id_disp(blocks_values,self.read_input_values_by_id(blocks_values,v2),v2)]] for v2 in vs if isinstance(vs,list) and len(vs) > 0]
                                
                            tr.append([self.get_opcode_from_id(blocks_values,ks),val])                        
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
    
    def count_non_opcodes(self,blocks_values,scratch_tree):
        non_opcodes = self.iterate_tree_for_non_opcodes2(scratch_tree,blocks_values)
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
        self.scratch_stats = {"number_of_nodes": nodes_val, "number_of_edges" : self.print_tree_top(blocks_values,file_name),"opcodes_statistics":opcode_tree,"non_opcodes_statistics":non_opcode_tree,"most_common_opcodes_statistics":most_common_opcode_tree,"most_common_non_opcodes_statistics":most_common_non_opcode_tree,"connections":flt,"all_nodes":self.get_all_nodes(blocks_values,scratch_tree,file_name)}
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

        for sublist in original_list:
            flat_sublist = [sublist[0]]

            for item in sublist[1:]:
                if isinstance(item, list):
                    self.flatten_inner_nested_lists(item)
                else:
                    if item not in flat_sublist:
                        flat_sublist.append(item)
            if flat_sublist not in flattened_list:
                flattened_list.append(flat_sublist)

        return flattened_list
    
    def convert_lst_to_nested_list(self,lst, current_path=[]):
        
        if isinstance(lst, list):
            #if lst[0] not in current_path:
            if lst[0] not in current_path:
                current_path.append(lst[0])

            if current_path not in self.final_list_result:
                self.final_list_result.append(current_path)
            
            if len(lst) > 1:
                for item in lst[1]:
                    self.convert_lst_to_nested_list(item,list(current_path))
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

    def read_files(self, parsed_file):
        self.parsed_value = self.sb3class.unpack_sb3(parsed_file)
        print("unpacked", {self.parsed_value})
        if len(self.parsed_value) > 0:
            self.blocs_json = json.loads(self.parsed_value)
        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
        
        #print(all_blocks_value)

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
        
        all_keys = self.get_all_keys(all_blocks_value)
        all = self.get_all(all_blocks_value,all_keys)
        
        non_opc = self.iterate_tree_for_non_opcodes2(next_val2,all_blocks_value)
        gp_tr = self.list_to_dict(next_val2)
        flt = self.flatten_tree(next_val2)
        #print(flt)
        #flt2 = self.convert_to_flat_list(next_val2)
        #print(flt2)
        root = list(gp_tr.keys())
        s =set()
        non_opcode = self.iterate_tree_for_non_opcodes2(next_val2,all_blocks_value)
        #print(non_opcode)
        opcode = self.return_all_opcodes(all_blocks_value)
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
        
        fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
        
               
        return fin_val

    def correct_parse(self,parsed_file):
        parsed_value = self.sb3class.unpack_sb3(parsed_file)
        if len(parsed_value) > 0:
            self.blocs_json = json.loads(parsed_value)
        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
        
        #print(all_blocks_value)

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]
        next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
        fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}

    def decode_scratch_bytes(self, raw_bytes):   
        
        with BytesIO(raw_bytes) as f:
            #self.scr_proj = f

            with tempfile.TemporaryFile(delete=False) as fp:
                fp.write(f)
            self.scr_proj = self.sb3class.unpack_sb3(fp.name)

        return self.scr_proj
    

    def decode2(self,raw_bytes,file_name):
        with BytesIO(raw_bytes) as f:
            self.scr_proj = self.sb3class.unpack_sb3(f)
            val = json.loads(self.scr_proj)
            all_blocks_value = self.get_all_blocks_vals(val)
            file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
            next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
            fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
            
            
            return fin_val
    def parse_scratch(self,scr_proj,file_name):
        
        if len(scr_proj) > 0:
            val = json.loads(scr_proj)
            all_blocks_value = self.get_all_blocks_vals(val)
            
            file_name = os.path.basename(file_name).split('/')[-1].split('.sb3')[0]
            next_val2 = self.create_next_values2_disp(all_blocks_value,file_name)
            fin_val = {"parsed_tree":next_val2,"stats":self.generate_summary_stats(all_blocks_value,file_name,next_val2)}
        
            return fin_val
        
    


        
#def main(filename: str):
    #pars = scratch_parser_inst.read_files(filename)   
    #return pars
        


#if __name__ == "__main__":
    #file_name = sys.argv[1]
    #main(file_name)

scratch_parser_inst = scratch_parser()
#print(scratch_parser_inst.read_files("files/Chicken Clicker remix-4.sb3"))

    
