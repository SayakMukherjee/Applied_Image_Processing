import json

class Config(object):

    def __init__(self, local_vars):
        self.local_vars = local_vars

    def load_config(self, import_path):

        with open(import_path, 'r') as file:
            local_vars = json.load(file)

        for key, val in  local_vars.items():
            self.local_vars[key] = val

    def save_config(self, export_path):

        with open(export_path, 'w') as file:
            json.dump(self.local_vars, file)
