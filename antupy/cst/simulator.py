"""
module for CST plants. CSP can be seen as a CST+ power plant

"""
from __future__ import annotations
from antupy.protocols import Simulation
from antupy.cst.optics import (
    SolarField,
    HyperboloidMirror,
    ParaboloidMirror
    )

from antupy.cst import spr
from antupy.cst.mcrt import TertiaryOpticalDevice



R0_COLS = ['xi','yi','zi', 'uxi','uyi','uzi', 'hel']
R1_COLS = ['hel','xi','yi','zi', 
           'xb','yb','zb', 'xc','yc','zc', 
           'uxi','uyi','uzi', 'uxb','uyb','uzb', 
           'hel_in', 'hit_hb', 'hit_tod']
R2_COLS = ['hel','xi','yi','zi', 
           'xb','yb','zb', 'xc','yc','zc', 
           'xs','ys','zs', 'xr','yr','zr', 
           'uxi','uyi','uzi', 'uxb','uyb','uzb', 
           'uxr','uyr','uzr', 
           'hel_in','hit_hb','hit_tod','hit_rcv','Nr_tod']


class CSTSimulation(Simulation):
    def __init__(self):
        self.name = "Concentrated Solar Thermal"
    
    solar_field = SolarField()
    hb =  HyperboloidMirror()
    tod = TertiaryOpticalDevice()
    receiver = spr.BlackboxModel()
    power_block : None
    storage : None

    def layout(self) -> None:
        return None
    
    def run_simulation(self) -> None:
        pass
    
