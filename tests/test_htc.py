import numpy as np

from antupy import Var
from antupy.htc import h_horizontal_surface_upper_hot

def test_horizontal_surface_upper_hot():
    T_s = 400.
    T_inf = 300.
    L = 1.
    assert (
        np.round(h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="Holman"), 1)
        == 7.1
        )
    assert (
        np.round(h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="NellisKlein"), 1)
        == 6.7
        )