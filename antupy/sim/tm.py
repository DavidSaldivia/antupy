"""
module for thermal simulations. In general, it takes a system (formed of devices) and timeseries (ts) and solve the thermal equations associated with the system using ts as constraints.
"""

class ThermalSimulation:
    def __init__(self):
        self.name = "Thermal Simulator"