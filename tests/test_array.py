import numpy as np
from antupy.core import Array, CF

def test_array_creation():
    data1 = [i for i in range(100)]
    data2 = np.arange(1,100)
    u1 = "m/s"
    u2 = "km/hr"
    cf = CF(u1,u2).v

    array1 = Array(data1, u1)
    array2 = Array(data2*cf, u2)
    assert array1 == array2
