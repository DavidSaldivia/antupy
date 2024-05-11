"""
module with the general settings for an analysis
"""

class Analyser():
    def __init__(self):
        self.name = "General Settings Analyser"

class PAnalyser(Analyser):
    def __init__(self):
        self.name = "Parametric Analyser"

class MCAnalyser(Analyser):
    def __init__(self):
        self.name = "MonteCarlo Analyser"

class LCAnalyser(Analyser):
    def __init__(self):
        self.name = "Life Cycle Assessment Analyser"