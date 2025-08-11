import numpy as np
from antupy.core import Array, Var, CF

def test_array_creation():
    data1 = [i for i in range(10)]
    data2 = np.arange(0,10)
    u1 = "m/s"
    u2 = "km/hr"
    cf = CF(u1,u2).v
    assert Array(data1, u1) == Array(data2, u1)
    assert Array(data1, u1) == Array(data2*cf, u2)
    assert Array(data1, u1) / cf == Array(data2, u2)

def test_add_array():
    array1 = Array([1,2,3], "day")
    array2 = Array([24,48,72], "hr")
    array3 = Array(np.array([2,4,6]),"day")
    assert array1 + array2 == array3
    assert array3 - array1 == array2

def test_mul_div_array():
    array1 = Array([1,2,3], "W")
    array2 = Array([4,5,6], "s")
    array3 = Array([4,10,18],"J")
    assert array1 * array2 == array3
    assert array3 / array2 == array1

def test_indexing_array():
    array = Array([1,2,3], "W")
    var1 = Var(1,"W")
    var2 = Var(2,"J/s")
    var3 = Var(3, "W")
    assert array[0] == var1
    assert array[1] == var2
    assert array[-1] == var3

def test_len_array():
    array = Array([1,2,3], "W")
    assert len(array) == 3

def test_iter_array():
    data = [1,2,3]
    u = "m"
    array = Array(data,u)
    i = 0
    for e in array:
        assert e == Var(data[i],u)
        i += 1
