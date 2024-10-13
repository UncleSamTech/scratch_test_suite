import json
import os
import zipfile

class unzip_scratch:

    def __init__(self):
        self.contents = ""

    def unpack_sb3(self,sb3_file,sprite=False):
        
        
        json_file = "project.json" 
        
        try:
            with zipfile.ZipFile(sb3_file) as sb3zip:
                if json_file in sb3zip.namelist():
                    # Read and decode the project.json file
                    self.contents = sb3zip.read(json_file).decode('utf-8')
                
                    # Parse the JSON content
                    loaded_json = json.loads(self.contents)
                
                    # Convert the loaded JSON back to a string (you can return the parsed object if needed)
                    return json.dumps(loaded_json)
                else:
                    print(f"{json_file} not found in {sb3_file}")
                return None
        except zipfile.BadZipFile:
            print("Error: Invalid .sb3 file.")
            return None
        
            
            