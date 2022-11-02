#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the base class for loading and saving experimental 
# configurations.
# ---------------------------------------------------------------------------

import json

class Config(object):
    """Base class for experimental configurations
    """

    def __init__(self, local_vars):

        self.local_vars = local_vars

    def load_config(self, import_path):
        """Load configurations from json file

        Args:
            import_path (str): path of config.json
        """

        with open(import_path, 'r') as file:
            local_vars = json.load(file)

        for key, val in  local_vars.items():
            self.local_vars[key] = val

    def save_config(self, export_path):
        """Save configurations to json file

        Args:
            export_path (str): path to export config.json
        """

        with open(export_path, 'w') as file:
            json.dump(self.local_vars, file)
