"""
module for storage units

"""

class ThermalStorage():
    def __init__(self):
        self.name = "Thermal Storage Unit"
        self.geometry = "cylinder"
        self.fluid = "solar_salt"


class Battery():
    def __init__(self):
        self.name = "Battery (Electrochemical storage)"


class CAES():
    def __init__(self):
        self.name = "Compressed Air Storage"


class ChemicalStorage():
    def __init__(self):
        self.name = "Chemical Storage"

