import os
import json


class DIRECTORY:
    DIR_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DIR_FILE = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(DIR_MAIN, ".dirs"), "r") as f:
        private_dirs = json.load(f)
    DIR_DATA_EXTERNAL = private_dirs["data"]
    FILE_TRNSYS_EXEC = private_dirs["trnsys"]
    DIR_TRNSYS_TEMP = private_dirs["trnsys_temp"]

    DIR_RESULTS = os.path.join(DIR_MAIN, "results")
    DIR_PROJECTS = os.path.join(DIR_MAIN, "projects")


class DEFINITIONS:
    SEASON = {
        "summer": [12, 1, 2],
        "autumn": [3, 4, 5],
        "winter": [6, 7, 8],
        "spring": [9, 10, 11],
    }
    MONTHS = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    DAYS_PER_MONTHS = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }
    DAYOFWEEK = {
        "weekday": [0, 1, 2, 3, 4],
        "weekend": [5, 6]
    }
    CLIMATE_ZONE = {
        1: "Hot humid summer",
        2: "Warm humid summer",
        3: "Hot dry summer, mild winter",
        4: "Hot dry summer, cold winter",
        5: "Warm summer, cool winter",
        6: "Mild warm summer, cold winter",
    }
    WEATHER_SIMULATIONS = {
        "annual" : "tmy",
        "mc": "mc",
        "historical": "historical",
        "hw_only" : "constant_day",
        "forecast": "mc",
    }


class DEFAULTS():
    #pv
    G_STC = 1000.                   #[W/m2]
    PV_NOMPOW = 5000.               #[W]
    ADR_PARAMS = {
        'k_a': 0.99924,
        'k_d': -5.49097,
        'tc_d': 0.01918,
        'k_rs': 0.06999,
        'k_rsh': 0.26144,
    }


class SIMULATIONS_IO():
    pass