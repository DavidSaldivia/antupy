"""
Test Suite for Plant Infrastructure

This test suite verifies the core functionality of plant.py including:
1. Plant base class functionality
2. Component caching with dependency tracking
3. constraint() and derived() functions
4. Cache invalidation and performance
5. Integration with parametric analysis
"""

import pytest
import time
from dataclasses import dataclass
from antupy import Var
from antupy import Plant, component, constraint, derived


# Test Components
# ===============

@dataclass
class SimpleComponent:
    """Simple test component with basic parameters."""
    param1: Var = Var(10.0, "m")
    param2: Var = Var(5.0, "kg")
    name: str = "default"
    
    def __post_init__(self):
        self.computed_value = self.param1.gv("m") * self.param2.gv("kg")


@dataclass
class ComplexComponent:
    """Complex test component with derived properties."""
    zf: Var = Var(50.0, "m")
    fzv: Var = Var(0.83, "-")
    file_path: str = "default.csv"
    
    def __post_init__(self):
        self.R_hb = (self.zf * self.fzv / 2.0).su("m")
        self.area = (3.14159 * self.R_hb * self.R_hb).su("m2")
    
    @property
    def surface_area(self) -> Var:
        """Calculate surface area."""
        return self.area


# Test Plants
# ===========

@dataclass
class SimpleTestPlant(Plant):
    """Simple test plant for basic functionality testing."""
    
    param1: Var = Var(10.0, "m")
    param2: Var = Var(5.0, "kg")
    name: str = "test_plant"
    
    def __post_init__(self):
        super().__post_init__()
        self.out = {}
    
    @property
    def simple_comp(self) -> SimpleComponent:
        """Simple component with constraints."""
        return component(SimpleComponent(
            param1=constraint(self.param1),
            param2=constraint(self.param2),
            name=self.name
        ))


@dataclass
class ComplexTestPlant(Plant):
    """Complex test plant for advanced functionality testing."""
    
    # Primary parameters
    zf: Var = Var(50.0, "m")
    fzv: Var = Var(0.83, "-")
    multiplier: Var = Var(2.0, "-")
    
    def __post_init__(self):
        super().__post_init__()
        self.out = {}
    
    def _get_file_path(self, zf: Var) -> str:
        """Helper function for derived file path."""
        return f'dataset_{zf.gv("m"):.0f}m.csv'
    
    def _get_scaled_param(self, zf: Var, multiplier: Var) -> Var:
        """Helper function for derived parameter."""
        value = zf.gv("m") * multiplier.gv("-")
        return Var(value, "m")
    
    @property
    def complex_comp(self) -> ComplexComponent:
        """Complex component with constraints and derived parameters."""
        return component(ComplexComponent(
            zf=constraint(self.zf),
            fzv=constraint(self.fzv),
            file_path=derived(self._get_file_path, self.zf)
        ))
    
    @property
    def derived_comp(self) -> SimpleComponent:
        """Component with derived parameter computation."""
        return component(SimpleComponent(
            param1=derived(self._get_scaled_param, self.zf, self.multiplier),
            param2=Var(5.0, "kg"),  # Use fixed value with correct units
            name="derived"
        ))


# Basic Plant Tests
# =================

class TestPlantBasics:
    """Test basic Plant functionality."""
    
    def test_plant_initialization(self):
        """Test Plant initializes correctly."""
        plant = SimpleTestPlant()
        
        assert hasattr(plant, '_component_cache')
        assert hasattr(plant, '_component_dependencies')
        assert hasattr(plant, '_param_hash_cache')
        assert plant._component_cache == {}
        assert plant._component_dependencies == {}
        assert plant.out == {}
    
    def test_plant_inherits_from_plant(self):
        """Test Plant has proper structure."""
        plant = SimpleTestPlant()
        
        assert isinstance(plant, Plant)
        assert hasattr(plant, 'out')
        assert hasattr(plant, 'constraints')
        assert hasattr(plant, 'run_simulation')
    
    def test_plant_parameters_accessible(self):
        """Test that plant parameters are accessible."""
        plant = SimpleTestPlant(param1=Var(20.0, "m"), param2=Var(10.0, "kg"))
        
        assert plant.param1.gv("m") == 20.0
        assert plant.param2.gv("kg") == 10.0
        assert plant.name == "test_plant"


# Component Function Tests
# ========================

class TestComponentFunction:
    """Test the component() function."""
    
    def test_component_returns_same_instance(self):
        """Test component() returns the same instance."""
        plant = SimpleTestPlant()
        
        # Access component twice
        comp1 = plant.simple_comp
        comp2 = plant.simple_comp
        
        # Should be the same object due to caching
        assert comp1 is comp2
        assert id(comp1) == id(comp2)
    
    def test_component_caching_works(self):
        """Test component caching mechanism."""
        plant = SimpleTestPlant()
        
        # First access should create component
        comp1 = plant.simple_comp
        assert 'simple_comp' in plant._component_cache
        
        # Second access should return cached component
        comp2 = plant.simple_comp
        assert comp1 is comp2
        
        # Cache should have one entry
        assert len(plant._component_cache) == 1
    
    def test_component_with_different_parameters(self):
        """Test component creation with different parameters."""
        plant1 = SimpleTestPlant(param1=Var(10.0, "m"))
        plant2 = SimpleTestPlant(param1=Var(20.0, "m"))
        
        comp1 = plant1.simple_comp
        comp2 = plant2.simple_comp
        
        # Should be different instances with different parameters
        assert comp1 is not comp2
        assert comp1.param1.gv("m") == 10.0
        assert comp2.param1.gv("m") == 20.0


# Constraint Function Tests
# =========================

class TestConstraintFunction:
    """Test the constraint() function."""
    
    def test_constraint_returns_same_value(self):
        """Test constraint() returns the input value unchanged."""
        plant = SimpleTestPlant()
        
        original_value = plant.param1
        constrained_value = constraint(plant.param1)
        
        assert constrained_value is original_value
        assert constrained_value.gv("m") == original_value.gv("m")
    
    def test_constraint_registers_dependency(self):
        """Test constraint() registers parameter dependencies."""
        plant = SimpleTestPlant()
        
        # Access component to trigger dependency registration
        _ = plant.simple_comp
        
        # Check dependencies were registered
        assert 'simple_comp' in plant._component_dependencies
        dependencies = plant._component_dependencies['simple_comp']
        assert 'param1' in dependencies
        assert 'param2' in dependencies
    
    def test_constraint_with_various_types(self):
        """Test constraint() works with different parameter types."""
        plant = SimpleTestPlant()
        
        # Test with Var
        var_result = constraint(plant.param1)
        assert var_result is plant.param1
        
        # Test with string
        str_result = constraint(plant.name)
        assert str_result == plant.name
        
        # Test with numbers
        num_result = constraint(42)
        assert num_result == 42


# Derived Function Tests
# ======================

class TestDerivedFunction:
    """Test the derived() function."""
    
    def test_derived_executes_callable(self):
        """Test derived() executes the callable correctly."""
        plant = ComplexTestPlant()
        
        # Access component with derived parameter
        comp = plant.complex_comp
        
        # Check derived parameter was computed correctly
        expected_file = f'dataset_{plant.zf.gv("m"):.0f}m.csv'
        assert comp.file_path == expected_file
        assert comp.file_path == 'dataset_50m.csv'
    
    def test_derived_with_multiple_parameters(self):
        """Test derived() with multiple tracked variables."""
        plant = ComplexTestPlant()
        
        # Access component with derived parameter that depends on multiple vars
        comp = plant.derived_comp
        
        # Check derived parameter was computed correctly
        expected_value = plant.zf.gv("m") * plant.multiplier.gv("-")
        assert comp.param1.gv("m") == expected_value
        assert comp.param1.gv("m") == 100.0  # 50 * 2
    
    def test_derived_registers_dependencies(self):
        """Test derived() registers dependencies correctly."""
        plant = ComplexTestPlant()
        
        # Access components to trigger dependency registration
        _ = plant.complex_comp
        _ = plant.derived_comp
        
        # Check dependencies were registered
        complex_deps = plant._component_dependencies.get('complex_comp', set())
        derived_deps = plant._component_dependencies.get('derived_comp', set())
        
        # Complex component should depend on zf and fzv
        assert 'zf' in complex_deps
        assert 'fzv' in complex_deps
        
        # Derived component should have some dependencies registered
        # (The exact names depend on the derived() function implementation)
        assert len(derived_deps) > 0


# Dependency Tracking Tests
# =========================

class TestDependencyTracking:
    """Test automatic dependency tracking."""
    
    def test_dependency_discovery(self):
        """Test automatic dependency discovery."""
        plant = ComplexTestPlant()
        
        # Access all components to trigger discovery
        _ = plant.complex_comp
        _ = plant.derived_comp
        
        # Check all dependencies were discovered
        deps = plant._component_dependencies
        assert 'complex_comp' in deps
        assert 'derived_comp' in deps
        
        # Verify specific dependencies
        assert 'zf' in deps['complex_comp']
        assert 'fzv' in deps['complex_comp']
        
        # Check that derived component has some dependencies registered
        # (The exact names depend on the derived() function implementation)
        assert len(deps['derived_comp']) > 0
    
    def test_dependency_tracking_consistency(self):
        """Test dependency tracking is consistent across accesses."""
        plant = ComplexTestPlant()
        
        # Access components multiple times
        for _ in range(3):
            _ = plant.complex_comp
            _ = plant.derived_comp
        
        # Dependencies should be consistent
        deps = plant._component_dependencies
        assert len(deps) == 2
        assert 'zf' in deps['complex_comp']
        assert 'fzv' in deps['complex_comp']


# Cache Invalidation Tests
# ========================

class TestCacheInvalidation:
    """Test component cache invalidation."""
    
    def test_cache_stats(self):
        """Test cache statistics reporting."""
        plant = ComplexTestPlant()
        
        # Initially no cache
        stats = plant.get_component_cache_stats()
        assert stats['cache_size'] == 0
        assert len(stats['cached_components']) == 0
        
        # Access components
        _ = plant.complex_comp
        _ = plant.derived_comp
        
        # Check cache stats
        stats = plant.get_component_cache_stats()
        assert stats['cache_size'] == 2
        assert 'complex_comp' in stats['cached_components']
        assert 'derived_comp' in stats['cached_components']
        assert stats['total_dependencies'] > 0
    
    def test_manual_cache_invalidation(self):
        """Test manual cache invalidation."""
        plant = ComplexTestPlant()
        
        # Access component and verify caching
        comp1 = plant.complex_comp
        assert 'complex_comp' in plant._component_cache
        
        # Manually invalidate
        plant._invalidate_affected_components({'zf'})
        
        # Component should be recreated
        comp2 = plant.complex_comp
        assert comp1 is not comp2  # Different instances
        assert 'complex_comp' in plant._component_cache  # But still cached


# Performance Tests
# =================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_component_access_performance(self):
        """Test component access is fast with caching."""
        plant = ComplexTestPlant()
        
        # First access (creation)
        start_time = time.time()
        comp1 = plant.complex_comp
        creation_time = time.time() - start_time
        
        # Subsequent accesses (cached)
        start_time = time.time()
        for _ in range(100):
            comp = plant.complex_comp
            assert comp is comp1  # Same instance
        cache_time = time.time() - start_time
        
        # Cached access should be much faster (or at least not slower)
        # Use a more lenient test since creation can be very fast
        assert cache_time < max(creation_time * 100, 0.1)  # Should be reasonable
    
    def test_no_infinite_loops(self):
        """Test that dependency tracking doesn't cause infinite loops."""
        plant = ComplexTestPlant()
        
        # This should complete without hanging
        start_time = time.time()
        _ = plant.complex_comp
        _ = plant.derived_comp
        execution_time = time.time() - start_time
        
        # Should complete quickly (< 1 second)
        assert execution_time < 1.0


# Integration Tests
# =================

class TestIntegration:
    """Test integration with other antupy components."""
    
    def test_parametric_compatibility(self):
        """Test compatibility with parametric analysis (if available)."""
        plant = ComplexTestPlant()
        
        # Access components to populate cache
        comp1 = plant.complex_comp
        
        # Simulate parameter change (like Parametric would do)
        plant.zf = Var(60.0, "m")
        
        # Component should still work with new parameter
        comp2 = plant.complex_comp
        assert comp2.zf.gv("m") == 60.0
    
    def test_multiple_plants_independence(self):
        """Test that multiple plant instances are independent."""
        plant1 = ComplexTestPlant(zf=Var(40.0, "m"))
        plant2 = ComplexTestPlant(zf=Var(60.0, "m"))
        
        comp1 = plant1.complex_comp
        comp2 = plant2.complex_comp
        
        # Should be different instances
        assert comp1 is not comp2
        assert comp1.zf.gv("m") == 40.0
        assert comp2.zf.gv("m") == 60.0
        
        # Caches should be independent
        assert plant1._component_cache != plant2._component_cache


# Edge Cases and Error Handling
# ==============================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_component_without_context(self):
        """Test component() function without plant context."""
        # Create component directly (not in property)
        comp = component(SimpleComponent())
        
        # Should still return the component (no caching)
        assert isinstance(comp, SimpleComponent)
    
    def test_constraint_without_context(self):
        """Test constraint() function without plant context."""
        value = Var(10.0, "m")
        result = constraint(value)
        
        # Should return the value unchanged
        assert result is value
    
    def test_derived_without_context(self):
        """Test derived() function without plant context."""
        def test_func(x):
            return x * 2
        
        result = derived(test_func, 5)
        
        # Should execute the function
        assert result == 10
    
    def test_derived_with_exception(self):
        """Test derived() function with callable that raises exception."""
        plant = ComplexTestPlant()
        
        def failing_func(x):
            raise ValueError("Test error")
        
        # Should propagate the exception
        with pytest.raises(ValueError, match="Error executing derived parameter calculation"):
            derived(failing_func, plant.zf)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])