import os
import json
from unzip_scratch import unzip_scratch

class scratch_parser:

    def __init__(self):
        
        self.blocs_json = None
        self.blocks_values = []
        self.sb3class = unzip_scratch()
        self.ommited_block_keys_parent = {"opcode"}
        self.all_opcodes = []
        self.scratch_tree_list = []
        self.scratch_tree = {}
        self.next_val_tree = {}
        self.input_block = {}
        self.sec_val = None
        self.in_val = None
        self.inpt_2 = []

    
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
    
        return block_values['blocks'][block_id]['opcode'] if block_values['blocks'][block_id]['opcode'] != None else ''
        
    def return_all_opcodes(self,blocks_values):
        return [v2['opcode'] for k,v in blocks_values.items() for v2 in v.values() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2) and 'opcode' in v2.keys()]

    def get_parent_opcode(self,blocks_values):
        if blocks_values == None or blocks_values == {}:
            return ''
        par = [v2['opcode'] for k,v in blocks_values.items() for v2 in v.values() if isinstance(v,dict) and bool(v) and isinstance(v2,dict) and bool(v2) and 'opcode' in v2.keys() and 'parent' in v2.keys() and v2["parent"] == None]
        return par[0] if len(par) == 1 else par
         

    def create_top_tree(self,block_values,next_values):
        if block_values == None or block_values == {}:
            return {}
        par_opcode = self.get_parent_opcode(block_values)
        if isinstance(par_opcode,list):
            for each_par in par_opcode:
                self.scratch_tree[each_par] = next_values
        else:
            self.scratch_tree[par_opcode] = next_values

            
        return self.scratch_tree 

    def create_top_tree2(self,block_values,next_values):
        if block_values == None or block_values == {}:
            return []
        par_opcode = self.get_parent_opcode(block_values)
        if isinstance(par_opcode,list):
            for each_par in par_opcode:
                self.scratch_tree_list = [each_par,next_values]
        else:
            self.scratch_tree_list = [par_opcode,next_values]
    
        return self.scratch_tree_list
    
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

    
    def flatten_input_values(self,blocks_values,id):
        if id == None or id == '' or blocks_values == None or blocks_values == {}:
            return {}
        input_block = self.read_input_values_by_id(blocks_values,id)
        print(input_block)
        if input_block == None or input_block == {}:
            return {}
        if isinstance(input_block,dict) and bool(input_block):
                for k,v in input_block.items():
                    if isinstance(v,list) and len(v) > 0:
                        if isinstance(v[1],list) and len(v[1]) > 0 and isinstance(v[1][1],str) and not isinstance(v[1],str):
                            self.input_block = {k:v[1][1]} if v[1][1] != '' or v[1][1] != None else {}
                                
                        elif isinstance(v[1],str) and len(v[1]) > 0 and not isinstance(v[1],list):
                            opcode = self.get_opcode_from_id(blocks_values,v[1])
                            block_by_id = self.get_any_block_by_id(blocks_values,v[1])
                            self.input_block = {k:{opcode:self.flatten_input_values(block_by_id,v[1])}}
                            
                for k2,v2 in input_block.items():
                    if k2 not in self.input_block.keys(): 
                        if isinstance(v2,list) and len(v2) > 0:
                            if isinstance(v2[1],str):
                                
                                opcode = self.get_opcode_from_id(blocks_values,v2[1])
                                
                                if opcode != None or opcode != '':
                                    val_flat = self.flatten_input_values(blocks_values,v2[1])
                                    
                                    if isinstance(val_flat,dict):
                                        self.sec_val = {opcode: val_flat} if self.check_dict_depth(val_flat) != 2 else {opcode:{ks:vs for ks,vs in val_flat.items()}}
                                        self.input_block = {k2:self.sec_val}
                                else:
                                    self.input_block = {k2:v2[1]}
                            elif not isinstance(v2[1],str) and isinstance(v2[1],list) and len(v2[1]) > 0 and isinstance(v2[1][1],str) and v2[1][1] != '' or v2[1][1] != None:
                                self.sec_val = v2[1][1]
                                self.input_block.update({k2:self.sec_val})

    def read_input_values(self,blocks_values,input_block):
        if input_block == None or input_block == {}:
            return {}
        if isinstance(input_block,dict) and bool(input_block):
            for k,v in input_block.items():
                if isinstance(v,list) and len(v) > 0:
                    for each_val in v:
                        if isinstance(each_val,str) and len(each_val) > 0 and each_val != '':
                            opcode = self.get_opcode_from_id(blocks_values,each_val)
                            new_inp_block = self.read_input_values_by_id(blocks_values,each_val)
                            
                            self.input_block = {k:{opcode:self.read_input_values(blocks_values,new_inp_block)}}
                        if isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0 and each_val[1] != '':
                            self.input_block = {k:each_val[1]}
            
            for k2,v2 in input_block.items():
                if k2 not in self.input_block.keys(): 
                    if isinstance(v2,list) and len(v2) > 0:
                        for each_val2 in v2:
                            if isinstance(each_val2,str) and len(each_val2) > 0 and each_val2 != '':
                                opcode2 = self.get_opcode_from_id(blocks_values,each_val2)
                                new_inp_block = self.read_input_values_by_id(blocks_values,each_val2)
                                self.input_block.update({k2:{opcode2:self.read_input_values(blocks_values,new_inp_block)}})
                            if isinstance(each_val2,list) and len(each_val2) > 0 and isinstance(each_val2[1],str) and len(each_val2[1]) > 0 and each_val2[1] != '':
                                self.input_block.update({k2:each_val2[1]})
                        
        return self.input_block

    def read_input_values2(self,blocks_values,input_block):
        if input_block == None or input_block == {} or blocks_values == None or blocks_values == {}:
            return []
        if isinstance(input_block,dict) and bool(input_block):
            for k,v in input_block.items():
                if isinstance(v,list) and len(v) > 0:
                    for each_val in v:
                        if isinstance(each_val,str) and len(each_val) > 0:
                            opcode = self.get_opcode_from_id(blocks_values,each_val)
                            new_inp_block = self.read_input_values_by_id(blocks_values,each_val) 
                            inp =self.read_input_values2(blocks_values,new_inp_block)
                            any_block = self.get_any_block_by_id(blocks_values,each_val)
                            next_opcode = self.get_opcode_from_id(blocks_values,any_block["next"])   
                            next_rec  = self.read_input_values2(blocks_values,self.read_input_values_by_id(blocks_values,any_block["next"]))  
                            
                            if inp == {} and  any_block["next"] == None:
                                
                                self.inpt_2 = [k,opcode]
                            
                            elif inp != {} and  any_block["next"] == None:
                                
                                if  inp in self.inpt_2 or opcode in self.inpt_2:
                                    continue
                                self.inpt_2 = [k,[opcode,[inp]]]
                            
                            elif inp != {} and any_block["next"] != None:
                                
                                if  next_rec in self.inpt_2 or next_opcode in self.inpt_2 or inp in self.inpt_2 or opcode in self.inpt_2:
                                    continue
                                self.inpt_2 = [k,[opcode,[inp],next_opcode,[next_rec]]]
                                #self.inpt_2 = [k,[opcode,[inp,[next_opcode,[next_rec]]]]]
                                
                        if isinstance(each_val,list) and len(each_val) > 0 and isinstance(each_val[1],str) and len(each_val[1]) > 0 and each_val[1] != '':
                            self.inpt_2 = [[k,each_val[1]]]

                          
            
            for k2,v2 in input_block.items():
                if k2 not in self.inpt_2: 
                    if isinstance(v2,list) and len(v2) > 0:
                        for each_val2 in v2:
                            if isinstance(each_val2,str) and len(each_val2) > 0:
                                opcode2 = self.get_opcode_from_id(blocks_values,each_val2)
                                new_inp_block = self.read_input_values_by_id(blocks_values,each_val2)
                                inp2 =self.read_input_values2(blocks_values,new_inp_block)
                                any_block2 = self.get_any_block_by_id(blocks_values,each_val2)
                                next_opcode2 = self.get_opcode_from_id(blocks_values,any_block2["next"])
                                next_rec2  = self.read_input_values2(blocks_values,self.read_input_values_by_id(blocks_values,any_block2["next"]))
                                
                                  
                                if inp2 == {} and any_block2["next"] == None:
                                    
                                    if  k2 in self.inpt_2 or opcode2 in self.inpt_2:
                                        continue
                                    self.inpt_2 = [k2,opcode2]
                                
                                elif  inp2 != {} and any_block2["next"] != None  :
                                    
                                    if  next_rec2 in self.inpt_2 or next_opcode2 in self.inpt_2 or inp2 in self.inpt_2 or opcode2 in self.inpt_2:
                                        continue
                    
                                    self.inpt_2 = [k2,[opcode2,[inp2],next_opcode2,[next_rec2]]]
                                    

                                elif inp2 != {} and any_block2["next"] == None:
                                    
                                    if  inp2 in self.inpt_2 or opcode2 in self.inpt_2:
                                        continue
                                    self.inpt_2 = [k2,[opcode2,[inp2]]]
                                    
                                
                            if isinstance(each_val2,list) and len(each_val2) > 0 and isinstance(each_val2[1],str) and len(each_val2[1]) > 0:
                                val = [k2,each_val2[1]]
                                if val in self.inpt_2:
                                    continue
                                self.inpt_2.append(val)
                                                              
        return self.inpt_2    
        
    def create_next_values(self,blocks_values):
        if blocks_values == None or blocks_values == {}:
            return {}
        val = {self.get_opcode_from_id(blocks_values,v):self.read_input_values(blocks_values,self.read_input_values_by_id(blocks_values,v)) for v in self.get_all_next_id(blocks_values) if isinstance(v,str) and v != ''}      
        return val
    
    def create_next_values2(self,blocks_values):   
        val = []
        if blocks_values == None or blocks_values == {}:
            return []
        for v in self.get_all_next_id(blocks_values):
            inp_by_id = self.read_input_values_by_id(blocks_values,v)
            
            inpval = self.read_input_values2(blocks_values,inp_by_id)
            if inpval in inpval[0:]:
                continue
            
        
            val.append([self.get_opcode_from_id(blocks_values,v),inpval])
        return val
    
    def get_children_keys(self,blocks_values):
        all_input_keys = []
        if blocks_values == None or blocks_values == {}:
            return []
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2,v2 in v.items():
                        if isinstance(v2,dict) and bool(v2) and 'inputs' in v2.keys():
                            v = v2['inputs']
                            for k3,v3 in v.items():
                                if isinstance(v3,list) and len(v3) > 0 and isinstance(v3[1],str) and len(v3[1]) > 0:
                                    all_input_keys.append(v3[1])
        return all_input_keys
    
    def get_next_child_keys(self,blocks_values):
        all_next_keys = []
        all_child_keys = self.get_children_keys(blocks_values)
        for each_key in all_child_keys:
            block = self.get_any_block_by_id(blocks_values,each_key)
            if isinstance(block,dict) and bool(block) and 'next' in block.keys():
                all_next_keys.append(block['next'])
        return all_next_keys


    def get_all_next_id(self,blocks_values):
        all_ids = []
        if blocks_values == None or blocks_values == {}:
            return {}
        if isinstance(blocks_values,dict) and bool(blocks_values):
            for k,v in blocks_values.items():
                if isinstance(v,dict) and bool(v):
                    for k2,v2 in v.items():
                        keyss = list(v.keys())
                        
                        for i in range(len(keyss)):
                            next_key_index = i + 1
                            
                            if next_key_index < len(keyss):
                                
                                if v2["next"] == keyss[next_key_index] and v2["next"] not in self.get_children_keys(blocks_values) and v2["next"] not in self.get_next_child_keys(blocks_values):
                                    all_ids.append(v2["next"])
                                elif v2["next"] != keyss[next_key_index] and v2["next"] == None:
                                    break                         
        return all_ids
    
    def read_files(self, parsed_file):
        self.parsed_value = self.sb3class.unpack_sb3(parsed_file)
        self.blocs_json = json.loads(self.parsed_value)
        
        


        #block values
        all_blocks_value = self.get_all_blocks_vals(self.blocs_json)
        
        #all opcodes
        all_opcodes = self.return_all_opcodes(all_blocks_value)
        
          
        inp_by_id = self.read_input_values_by_id(all_blocks_value,"Ml@l~n|$+$jrg9V%IzC{")
        #inp = self.read_input_values(all_blocks_value,inp_by_id)
        next_val = self.create_next_values(all_blocks_value)

        top_tree = self.create_top_tree(all_blocks_value,next_val)

        file_name = os.path.basename(parsed_file).split('/')[-1].split('.sb3')[0]

        #with open(f"files/{file_name}_tree2.json","w") as tree_file:
            #json.dump(top_tree,tree_file,indent=4)
        #print(top_tree)
        next_val2 = self.create_next_values2(all_blocks_value)
        
        top_tree2 = self.create_top_tree2(all_blocks_value,next_val2)
        print(top_tree2)  
        
        
        

scratch_parser_inst = scratch_parser()
scratch_parser_inst.read_files("files/3l_opcode.sb3")

    
