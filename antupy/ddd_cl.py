import os
import json


class DIRECTORY:

    DIR_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DIR_FILE = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(DIR_MAIN, ".dirs"), "r") as f:
        private_dirs = json.load(f)
    DIR_DATA = private_dirs["data"]


class DEFINITIONS:
    pass

class DEFAULTS:
    pass
