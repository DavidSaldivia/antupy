from antupy.protos import Model

class CSP():
    def __init__(self):
        self.name = "Concentrated Solar Power Plant"

# local protocols
class OpticalSubsystem(Model):
    pass

class ThermalSubsystem(Model):
    pass

class PowerSubsystem(Model):
    pass


# module classes
class BeamDownReceiver(OpticalSubsystem):
    pass