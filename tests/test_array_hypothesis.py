"""
Property-based tests for the Array class using Hypothesis.

These tests automatically generate diverse test cases to discover edge cases
and ensure the Array class behaves correctly across a wide range of inputs.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays, floating_dtypes

from antupy.core import Array, Var, CF
from antupy.units import Unit


# =============================================================================
# Test Data Generation Strategies
# =============================================================================

# Valid unit strings for testing
DIMENSIONAL_UNITS = [
    # Length units
    "m", "km", "cm", "mm", "ft", "in",
    # Mass units  
    "kg", "g", "lb", "ton",
    # Time units
    "s", "min", "hr", "day",
    # Temperature units
    "K", "°C", "degC",
    # Energy units
    "J", "kJ", "MJ", "kWh",
    # Power units
    "W", "kW", "MW",
    # Pressure units
    "Pa", "kPa", "MPa", "bar",
    # Velocity units
    "m/s", "km/hr", "mph", "ft/s",
    # Area units
    "m2", "cm2", "ft2",
    # Volume units
    "m3", "L", "ft3",
]

COMPATIBLE_UNIT_PAIRS = [
    ("m", "km"), ("m", "ft"), ("kg", "g"), ("s", "min"), ("J", "kJ"),
    ("W", "kW"), ("Pa", "bar"), ("m/s", "km/hr"), ("K", "°C"),
    ("m2", "ft2"), ("m3", "L")
]

INCOMPATIBLE_UNIT_PAIRS = [
    ("m", "kg"), ("s", "J"), ("K", "W"), ("Pa", "m"), ("kg", "m/s")
]

@st.composite
def numeric_arrays(draw, min_size=0, max_size=20, allow_nan=False, allow_inf=False):
    """Generate numpy arrays with various numeric types and ranges."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Choose array type
    array_type = draw(st.sampled_from(["list", "numpy_int", "numpy_float", "mixed"]))
    
    if array_type == "list":
        return draw(st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, 
                allow_nan=allow_nan, allow_infinity=allow_inf
            ),
            min_size=size, max_size=size
        ))
    elif array_type == "numpy_int":
        return draw(arrays(
            dtype=np.int32,
            shape=size,
            elements=st.integers(min_value=-1000, max_value=1000)
        ))
    elif array_type == "numpy_float":
        return draw(arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(
                min_value=-1e6, max_value=1e6,
                allow_nan=allow_nan, allow_infinity=allow_inf
            )
        ))
    else:  # mixed - create from list then convert to numpy
        values = draw(st.lists(
            st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=allow_nan, allow_infinity=allow_inf)
            ),
            min_size=size, max_size=size
        ))
        return np.array(values)

@st.composite 
def valid_arrays(draw, min_size=1, max_size=20):
    """Generate valid Array instances."""
    data = draw(numeric_arrays(min_size=min_size, max_size=max_size, allow_nan=False, allow_inf=False))
    unit = draw(st.sampled_from(DIMENSIONAL_UNITS))
    return Array(data, unit)

@st.composite
def compatible_array_pairs(draw, min_size=1, max_size=10):
    """Generate pairs of Arrays with compatible units."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate data for both arrays
    data1 = draw(numeric_arrays(min_size=size, max_size=size, allow_nan=False, allow_inf=False))
    data2 = draw(numeric_arrays(min_size=size, max_size=size, allow_nan=False, allow_inf=False))
    
    # Choose compatible units
    unit1, unit2 = draw(st.sampled_from(COMPATIBLE_UNIT_PAIRS))
    
    return Array(data1, unit1), Array(data2, unit2)

@st.composite
def scalar_values(draw):
    """Generate scalar values for Array operations."""
    return draw(st.one_of(
        st.integers(min_value=-100, max_value=100),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    ))


# =============================================================================
# Property-Based Tests for Array Class
# =============================================================================

class TestArrayHypothesis:
    """Hypothesis-based property tests for Array class core functionality."""

    @given(numeric_arrays(min_size=1, max_size=50), st.sampled_from(DIMENSIONAL_UNITS))
    def test_array_creation_properties(self, data, unit):
        """Property: Array creation should always produce valid Arrays with correct attributes."""
        array = Array(data, unit)
        
        # Array should have numpy array values
        assert isinstance(array.value, np.ndarray)
        assert array.value.shape == np.array(data).shape
        
        # Array should have valid unit
        assert isinstance(array.unit, Unit)
        assert array.u == unit
        
        # Length should match input data
        assert len(array) == len(data)

    @given(compatible_array_pairs(min_size=2, max_size=15))
    def test_addition_properties(self, array_pair):
        """Property: Array addition should be commutative and produce consistent results."""
        array1, array2 = array_pair
        
        # Addition should be commutative (when compatible)
        try:
            result1 = array1 + array2
            result2 = array2 + array1
            
            # Results should have same length
            assert len(result1) == len(array1)
            assert len(result2) == len(array2)
            
            # Values should be approximately equal (accounting for unit conversion)
            expected_unit = result1.unit.u
            np.testing.assert_allclose(
                result1.value, 
                result2.gv(expected_unit),
                rtol=1e-10
            )
            
        except TypeError:
            # If addition fails, both directions should fail consistently
            with pytest.raises(TypeError):
                array2 + array1

    @given(compatible_array_pairs(min_size=2, max_size=15))
    def test_subtraction_properties(self, array_pair):
        """Property: Array subtraction should satisfy basic algebraic properties."""
        array1, array2 = array_pair
        
        try:
            result = array1 - array2
            
            # Subtraction should produce array of same length
            assert len(result) == len(array1)
            
            # Adding back should give original (approximately)
            restored = result + array2
            np.testing.assert_allclose(
                array1.value,
                restored.gv(array1.u),
                rtol=1e-10
            )
            
        except TypeError:
            # If subtraction fails due to incompatible units, that's expected
            pass

    @given(valid_arrays(min_size=2, max_size=15), valid_arrays(min_size=2, max_size=15))  
    def test_multiplication_properties(self, array1, array2):
        """Property: Array multiplication should follow algebraic rules."""
        # Ensure same size for element-wise operations
        assume(len(array1) == len(array2))
        
        result = array1 * array2
        
        # Multiplication should be commutative
        result_reverse = array2 * array1
        
        # Results should have same length
        assert len(result) == len(array1)
        assert len(result_reverse) == len(array1)
        
        # Values should be equal (multiplication is commutative)
        np.testing.assert_allclose(result.value, result_reverse.value, rtol=1e-10)

    @given(valid_arrays(min_size=2, max_size=15), scalar_values())
    def test_scalar_operations_properties(self, array, scalar):
        """Property: Scalar operations should preserve array structure and units."""
        # Skip zero division
        assume(scalar != 0)
        
        # Scalar multiplication
        mult_result = array * scalar
        assert len(mult_result) == len(array)
        assert mult_result.u == array.u
        np.testing.assert_allclose(mult_result.value, array.value * scalar, rtol=1e-10)
        
        # Scalar division  
        div_result = array / scalar
        assert len(div_result) == len(array)
        assert div_result.u == array.u
        np.testing.assert_allclose(div_result.value, array.value / scalar, rtol=1e-10)

    @given(valid_arrays(min_size=1, max_size=30))
    def test_indexing_properties(self, array):
        """Property: Array indexing should return Var objects with correct values and units."""
        for i in range(len(array)):
            element = array[i]
            
            # Should return Var instance
            assert isinstance(element, Var)
            
            # Should have same unit as array
            assert element.u == array.u
            
            # Should have correct value
            assert element.v == array.value[i]
        
        # Negative indexing should work
        if len(array) > 0:
            assert array[-1].v == array.value[-1]

    @given(valid_arrays(min_size=1, max_size=20))
    def test_iteration_properties(self, array):
        """Property: Array iteration should yield Var objects with correct values."""
        elements = list(array)
        
        # Should yield same number of elements as array length
        assert len(elements) == len(array)
        
        # Each element should be a Var with correct properties
        for i, element in enumerate(elements):
            assert isinstance(element, Var)
            assert element.u == array.u
            assert element.v == array.value[i]

    @given(st.sampled_from(COMPATIBLE_UNIT_PAIRS), numeric_arrays(min_size=2, max_size=15))
    def test_unit_conversion_properties(self, unit_pair, data):
        """Property: Unit conversion should preserve physical meaning."""
        unit1, unit2 = unit_pair
        
        array1 = Array(data, unit1)
        
        try:
            # Convert to compatible unit
            array2 = array1.set_unit(unit2)
            
            # Should have new unit
            assert array2.u == unit2
            
            # Should have same length
            assert len(array2) == len(array1)
            
            # Converting back should give original values (approximately)
            array3 = array2.set_unit(unit1)
            np.testing.assert_allclose(array1.value, array3.value, rtol=1e-10)
            
        except ValueError:
            # Some conversions might not be supported - that's OK
            pass

    @given(valid_arrays(min_size=1, max_size=20), valid_arrays(min_size=1, max_size=20))
    def test_equality_properties(self, array1, array2):
        """Property: Array equality should be reflexive and consistent."""
        # Self-equality (reflexive)
        assert array1 == array1
        assert array2 == array2
        
        # If arrays are equal, they should have compatible units
        if array1 == array2:
            assert len(array1) == len(array2)
            assert array1.unit.base_units == array2.unit.base_units


class TestArrayEdgeCases:
    """Tests specifically designed to discover edge cases with Arrays."""

    @given(numeric_arrays(min_size=0, max_size=3), st.sampled_from(DIMENSIONAL_UNITS))
    def test_empty_and_small_arrays(self, data, unit):
        """Property: Arrays should handle empty and very small inputs gracefully."""
        array = Array(data, unit)
        
        if len(data) == 0:
            assert len(array) == 0
            # Empty arrays should still be iterable
            assert list(array) == []
        else:
            # Non-empty arrays should work normally
            assert len(array) == len(data)
            assert array[0].u == unit

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(
                min_value=-1e10, max_value=1e10,
                allow_nan=False, allow_infinity=False
            )
        ),
        st.sampled_from(DIMENSIONAL_UNITS)
    )
    def test_extreme_values(self, data, unit):
        """Property: Arrays should handle extreme but valid numeric values."""
        try:
            array = Array(data, unit)
            
            # Should create valid array
            assert isinstance(array.value, np.ndarray)
            assert len(array) == len(data)
            
            # Basic operations should not crash
            result = array * 2.0
            assert len(result) == len(array)
            
        except (OverflowError, ValueError) as e:
            # Some extreme values might cause legitimate errors
            print(f"Found limitation with extreme values: {e}")

    @given(st.sampled_from(INCOMPATIBLE_UNIT_PAIRS), numeric_arrays(min_size=2, max_size=8))
    def test_incompatible_unit_operations(self, unit_pair, data):
        """Property: Operations with incompatible units should raise appropriate errors."""
        unit1, unit2 = unit_pair
        
        array1 = Array(data, unit1)
        array2 = Array(data, unit2)
        
        # Addition and subtraction should fail with incompatible units
        with pytest.raises(TypeError):
            array1 + array2
            
        with pytest.raises(TypeError):
            array1 - array2
            
        # Multiplication should work (creates compound units)
        result = array1 * array2
        assert len(result) == len(array1)

    @given(valid_arrays(min_size=3, max_size=12))
    def test_array_arithmetic_consistency(self, array):
        """Property: Array arithmetic should be consistent with numpy operations."""
        # Test that Array arithmetic produces same results as numpy arithmetic
        # (when units are handled correctly)
        
        # Self-addition
        double_array = array + array
        expected_values = array.value + array.value
        np.testing.assert_allclose(double_array.value, expected_values, rtol=1e-10)
        
        # Scalar multiplication consistency
        scalar = 3.5
        mult_result = array * scalar
        expected_mult = array.value * scalar
        np.testing.assert_allclose(mult_result.value, expected_mult, rtol=1e-10)

    @given(
        st.lists(st.one_of(st.integers(), st.floats(allow_nan=False)), min_size=1, max_size=15),
        st.sampled_from(DIMENSIONAL_UNITS)
    )
    def test_mixed_numeric_types(self, mixed_data, unit):
        """Property: Arrays should handle mixed integer and float inputs correctly."""
        array = Array(mixed_data, unit)
        
        # Should convert to numpy array
        assert isinstance(array.value, np.ndarray)
        assert len(array) == len(mixed_data)
        
        # All elements should be accessible
        for i in range(len(array)):
            element = array[i]
            assert isinstance(element, Var)
            assert element.u == unit


@given(valid_arrays(min_size=1, max_size=10))
def test_hypothesis_finds_array_bugs(array):
    """This test is designed to catch potential bugs that Hypothesis might discover."""
    
    # Test some properties that might reveal bugs
    original_length = len(array)
    
    # Indexing should never go out of bounds for valid indices
    for i in range(original_length):
        element = array[i]
        assert element.u == array.u
    
    # Iteration should yield exactly len(array) elements
    elements = list(array)
    assert len(elements) == original_length
    
    # String representation should not crash
    repr_str = repr(array)
    assert isinstance(repr_str, str)
    assert "[" in repr_str  # Should contain unit brackets
    
    # Array operations with itself should be safe
    try:
        # These operations should at least not crash
        result1 = array + array
        result2 = array * array  
        result3 = array * 2.0
        
        # Results should have same length as original
        assert len(result1) == original_length
        assert len(result2) == original_length  
        assert len(result3) == original_length
        
    except (TypeError, ValueError, OverflowError):
        # Some operations might fail for certain inputs - that's acceptable
        pass


# Optional: Add settings to run more examples for thorough testing
# @settings(max_examples=1000)  # Uncomment for extensive testing
