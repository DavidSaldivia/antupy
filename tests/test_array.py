import numpy as np
from antupy.core import Array, CF

def test_array_creation():
    data1 = [i for i in range(10)]
    data2 = np.arange(0,10)
    u1 = "m/s"
    u2 = "km/hr"
    cf = CF(u1,u2).v
    assert Array(data1, u1) == Array(data2, u1)
    assert Array(data1, u1) == Array(data2*cf, u2)
