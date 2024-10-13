import json
import os
import zipfile

class unzip_scratch:

    def __init__(self):
        self.contents = ""

    def unpack_sb3(self,sb3_file,sprite=False):
        
        #file_name = os.path.basename(sb3_file).split('/')[-1].split('.sb3')[0]
        json_file = "project.json" if sprite else "project.json"
        
        with zipfile.ZipFile(sb3_file) as sb3zip:
            names = sb3zip.namelist()
            print(names)
            
            if json_file in names:
                
                self.contents += sb3zip.read(json_file).decode('utf-8')
                
                loaded_json = json.loads(self.contents)
                adv = json.dumps(loaded_json)
                
                return adv
        
            
            