import numpy as np
from antupy import Array, Var, CF

def test_array_creation():
    data1 = [i for i in range(10)]
    data2 = np.arange(0,10)
    u1 = "m/s"
    u2 = "km/hr"
    cf = CF(u1,u2).v
    assert Array(data1, u1) == Array(data2, u1)
    assert Array(data1, u1) == Array(data2*cf, u2)
    assert Array(data1, u1) / cf == Array(data2, u2)

def test_init_array():
    data = [1., 2., 3.]
    arr1 = Array(data, "m")
    arr2 = Array(arr1)
    arr3 = Array(arr1, "m")
    arr4 = Array(arr1, "km")

    assert isinstance(arr2, Array)
    assert isinstance(arr3, Array)
    assert isinstance(arr4, Array)

def test_array_var_operations():
    array = Array([1,2,3], "m")
    var = Var(2, "m")
    assert array + var == Array([3,4,5], "m")
    assert array - var == Array([-1,0,1], "m")
    assert var + array == Array([3,4,5], "m")
    assert var - array == Array([1,0,-1], "m")
    assert array * var == Array([2,4,6], "m2")
    assert var * array == Array([2,4,6], "m2")
    assert array / var == Array([0.5,1,1.5], "")
    assert var / array == Array([2,1,2/3], "")


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

# --- unit conversion ---

def test_gv_unit_conversion():
    array = Array([1.0, 2.0, 3.0], "km")
    result = array.gv("m")
    assert np.allclose(result, [1000.0, 2000.0, 3000.0])

def test_su_unit_conversion():
    array = Array([1.0, 2.0], "km")
    converted = array.su("m")
    assert converted == Array([1000.0, 2000.0], "m")

def test_su_incompatible_raises():
    import pytest
    array = Array([1.0, 2.0], "m")
    with pytest.raises(ValueError):
        array.su("kg")

def test_compatible():
    array = Array([1.0], "m")
    result = array.compatible()
    assert isinstance(result, list)
    assert "ft" in result

# --- statistical methods ---

def test_array_mean():
    array = Array([2.0, 4.0, 6.0], "m")
    assert array.mean() == Var(4.0, "m")

def test_array_mean_with_unit():
    array = Array([1000.0, 2000.0, 3000.0], "m")
    result = array.mean("km")
    assert result == Var(2.0, "km")

def test_array_std():
    array = Array([2.0, 4.0, 6.0], "m")
    assert array.std() == Var(float(np.std([2.0, 4.0, 6.0])), "m")

def test_array_max_min():
    array = Array([1.0, 5.0, 3.0], "m")
    assert array.max() == Var(5.0, "m")
    assert array.min() == Var(1.0, "m")

def test_array_argmax_argmin():
    array = Array([1.0, 5.0, 3.0], "m")
    assert array.argmax() == 1
    assert array.argmin() == 0

def test_array_sum():
    array = Array([1.0, 2.0, 3.0], "m")
    assert array.sum() == Var(6.0, "m")

def test_array_cumsum():
    array = Array([1.0, 2.0, 3.0], "m")
    assert array.cumsum() == Array([1.0, 3.0, 6.0], "m")

def test_array_round():
    array = Array([1.234, 2.567], "m")
    assert array.round(1) == Array([1.2, 2.6], "m")

def test_array_var():
    array = Array([2.0, 4.0, 6.0], "m")
    expected = np.var([2.0, 4.0, 6.0])
    result = array.var()
    assert abs(result.gv("m2") - expected) < 1e-10

# --- repr ---

def test_array_repr():
    array = Array([1.0, 2.0], "m")
    r = repr(array)
    assert "m" in r

# --- compatible-unit arithmetic branches ---

def test_add_compatible_units():
    a = Array([1.0, 2.0], "km")
    b = Array([500.0, 1000.0], "m")
    result = a + b
    assert result == Array([1.5, 3.0], "km")

def test_sub_compatible_units():
    a = Array([2.0, 4.0], "km")
    b = Array([500.0, 1000.0], "m")
    result = a - b
    assert result == Array([1.5, 3.0], "km")

def test_radd_compatible_units():
    a = Array([1.0, 2.0], "km")
    b = Var(500.0, "m")
    result = b + a
    assert result == Array([1.5, 2.5], "km")

def test_rsub_compatible_units():
    a = Array([0.5, 1.0], "km")
    b = Var(2000.0, "m")
    result = b - a
    assert result == Array([1.5, 1.0], "km")

def test_scalar_mul():
    array = Array([1.0, 2.0, 3.0], "m")
    result = array * 2.0
    assert result == Array([2.0, 4.0, 6.0], "m")

def test_scalar_rmul():
    array = Array([1.0, 2.0, 3.0], "m")
    result = 3.0 * array
    assert result == Array([3.0, 6.0, 9.0], "m")

def test_scalar_div():
    array = Array([2.0, 4.0, 6.0], "m")
    result = array / 2.0
    assert result == Array([1.0, 2.0, 3.0], "m")

def test_add_incompatible_raises():
    import pytest
    a = Array([1.0], "m")
    b = Array([1.0], "kg")
    with pytest.raises(TypeError):
        _ = a + b

def test_sub_incompatible_raises():
    import pytest
    a = Array([1.0], "m")
    b = Array([1.0], "kg")
    with pytest.raises(TypeError):
        _ = a - b

def test_eq_none_value():
    array = Array([1.0, 2.0], "m")
    other = Array(None, "m")
    assert (array == other) is False
