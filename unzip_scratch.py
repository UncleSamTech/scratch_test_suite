import json
import os
import zipfile

class unzip_scratch:

    def __init__(self):
        pass

    def unpack_sb3(self,sb3_file,sprite=False):
        json_file = "project.json" if sprite else "project.json"
        with zipfile.ZipFile(sb3_file) as sb3zip:
            names = sb3zip.namelist()
            
            if json_file in names:
                
                get_zipped_content_bytes = sb3zip.read(json_file).decode("utf-8")
                loaded_json = json.loads(get_zipped_content_bytes)
                print("value lodaded json ",loaded_json)
                return json.dumps(loaded_json)
            else:
                return ""
            