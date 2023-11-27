import json
import os
import zipfile

class unzip_scratch:

    def __init__(self):
        pass

    def unpack_sb3(self,sb3_file,sprite=False):
        need = "project.json" if sprite else "project.json"
        with zipfile.ZipFile(sb3_file) as sb3zip:
            names = sb3zip.namelist()
            
            if need not in names:
                print('sprite3 must contain sprite.json')
            else:
                json_file =  json.loads(sb3zip.read(need).decode("utf-8"))
                return json.dumps(json_file)