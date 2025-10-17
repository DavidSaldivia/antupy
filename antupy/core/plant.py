"""
Smart Plant Infrastructure for antupy

This module provides intelligent component management for Plant classes,
enabling automatic caching and selective invalidation of components based
on parameter dependencies.
"""

from typing import Any, Set, get_type_hints, TypeVar, Callable, TYPE_CHECKING, Generic, Type
from dataclasses import dataclass, field
import inspect
import hashlib

# Import only what we need to avoid circular imports
if TYPE_CHECKING:
    from antupy import Var, Array

# Type variable for component decorator
T = TypeVar('T')


@dataclass
class Plant():
    """
    Enhanced Plant base class with intelligent component management.
    
    Features:
    - Automatic component recreation on parameter changes
    - Dependency tracking between parameters and components  
    - Efficient caching with hash-based invalidation
    - Dynamic parameter introspection
    
    Usage:
        class MyPlant(Plant):
            # Define parameters as usual
            param1: Var = Var(10.0, "m")
            param2: Var = Var(5.0, "kg")
            
            @property
            def my_component(self) -> SomeComponentClass:
                return component(SomeComponentClass(
                    param1=constraint(self.param1),
                    param2=constraint(self.param2)
                ))
                
            @property 
            def complex_component(self) -> AnotherComponentClass:
                return component(AnotherComponentClass(
                    param1=constraint(self.param1),
                    computed_param=derived(self._compute_something, self.param1)
                ))
    """
    
    # Base Plant attributes (preserved for Protocol compatibility)
    out: dict[str, 'Var|Array|float'] = field(default_factory=dict)
    constraints: list[tuple[str, ...]] = field(default_factory=list)
    
    # Component registry: maps component names to their classes and dependencies
    _component_dependencies: dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Component instances cache
    _component_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Parameter hash cache for change detection
    _param_hash_cache: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Track current property context for component() function
    _current_property_context: str | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        # Initialize Plant-specific attributes (preserved for Protocol compatibility)
        if not hasattr(self, 'out'):
            self.out = {}
        # Note: dependency discovery will happen lazily when first component is accessed
    
    def run_simulation(self, verbose: bool = False) -> dict[str, 'Var|Array|float']:
        """Run simulation method (preserved for Protocol compatibility)."""
        # This method should be implemented by subclasses
        # Default implementation returns current out dict
        return self.out
    
    def __getattribute__(self, name):
        """Override to track property access for component() function context."""
        # Get the attribute normally
        attr = object.__getattribute__(self, name)
        
        # Check if this is a property access and we're accessing a component property
        cls = object.__getattribute__(self, '__class__')
        class_attr = getattr(cls, name, None)
        
        if isinstance(class_attr, property):
            # Set context so component() function knows which property is being accessed
            object.__setattr__(self, '_current_property_context', name)
            try:
                # Call the property getter - this should be the actual property call
                if class_attr.fget is not None:
                    result = class_attr.fget(self)  # Call property function directly
                    return result
                else:
                    return attr
            finally:
                # Clear context after property access
                object.__setattr__(self, '_current_property_context', None)
        
        return attr
    
    def _ensure_dependencies_discovered(self):
        """Ensure component dependencies have been discovered (lazy initialization)."""
        if not self._component_dependencies:
            self._discover_component_dependencies()
    
    def _discover_component_dependencies(self):
        """Auto-discover which parameters affect which components."""
        # Clear any existing dependencies first
        self._component_dependencies.clear()
        
        # Force component access to trigger dependency discovery
        for name in dir(self):
            attr = getattr(type(self), name, None)
            if isinstance(attr, property):
                try:
                    # Access the property to trigger component() calls and dependency registration
                    # This will populate _component_dependencies through constraint() and derived() calls
                    _ = getattr(self, name)
                except:
                    # If property access fails, skip this component
                    continue
    
    def _get_component_parameters(self, component_class) -> set[str]:
        """Extract parameter names from component __init__ signature."""
        sig = inspect.signature(component_class.__init__)
        return {
            param_name for param_name, param in sig.parameters.items()
            if param_name != 'self'
        }
    
    def _get_computed_parameters(self, component_name: str) -> set[str]:
        """Find parameters used in computed parameter methods."""
        computed_params = set()
        
        # Look for _get_* methods that this component might use
        for attr_name in dir(self):
            if attr_name.startswith('_get_'):
                computed_params.add(attr_name[4:])  # Remove '_get_' prefix
        
        return computed_params
    
    def _compute_params_hash(self, param_names: set[str]) -> str:
        """Compute hash of current parameter values."""
        param_values = []
        for param_name in sorted(param_names):  # Sort for consistent hashing
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                if hasattr(value, 'gv'):  # Var object
                    param_values.append(f"{param_name}={value.gv()}|{value.unit}")
                else:
                    param_values.append(f"{param_name}={value}")
        
        param_string = "|".join(param_values)
        hash_val = hashlib.md5(param_string.encode()).hexdigest()
        return hash_val
    
    def _invalidate_affected_components(self, changed_params: set[str]):
        """
        Invalidate only components affected by changed parameters.
        
        This is the key method that should be called by Parametric._update_parameters()
        instead of __post_init__() to enable smart component caching.
        
        Args:
            changed_params: Set of parameter names that have changed
        """
        self._ensure_dependencies_discovered()
        
        for component_name, dependencies in self._component_dependencies.items():
            if changed_params & dependencies:  # Intersection check
                # Clear component from cache - will be recreated on next access
                if component_name in self._component_cache:
                    del self._component_cache[component_name]
                if component_name in self._param_hash_cache:
                    del self._param_hash_cache[component_name]
    
    def _needs_component_recreation(self, component_name: str) -> bool:
        """Check if component needs recreation."""
        # Ensure dependencies are discovered
        self._ensure_dependencies_discovered()
        
        if component_name not in self._component_cache:
            return True
        
        if component_name not in self._component_dependencies:
            return True
        
        # Check parameter hash
        dependencies = self._component_dependencies[component_name]
        current_hash = self._compute_params_hash(dependencies)
        cached_hash = self._param_hash_cache.get(component_name)
        
        return current_hash != cached_hash
    
    def _update_component_hash(self, component_name: str):
        """Update the parameter hash for a component."""
        if component_name in self._component_dependencies:
            dependencies = self._component_dependencies[component_name]
            current_hash = self._compute_params_hash(dependencies)
            self._param_hash_cache[component_name] = current_hash
    
    def _build_component_kwargs(self, component_class):
        """Build kwargs for component, with smart parameter mapping."""
        sig = inspect.signature(component_class.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if hasattr(self, param_name):
                kwargs[param_name] = getattr(self, param_name)
            elif hasattr(self, f'_get_{param_name}'):
                # Call computed parameter method
                kwargs[param_name] = getattr(self, f'_get_{param_name}')()
            elif param.default != inspect.Parameter.empty:
                # Use default value if available
                kwargs[param_name] = param.default
            else:
                # Required parameter not found
                raise TypeError(f"Required parameter '{param_name}' not found for component {component_class.__name__}")
                
        return kwargs
    
    def get_component_cache_stats(self) -> dict[str, Any]:
        """Get statistics about component cache usage."""
        return {
            'cached_components': list(self._component_cache.keys()),
            'component_dependencies': dict(self._component_dependencies),
            'cache_size': len(self._component_cache),
            'total_dependencies': sum(len(deps) for deps in self._component_dependencies.values())
        }


def component(instance: T, component_name: str | None = None) -> T:
    """
    Cache-aware component function for use within @property methods.
    
    This function works alongside constraint() and derived() to provide:
    1. Component instance pass-through (returns the input instance)
    2. Smart caching based on parameter dependencies
    3. Automatic cache invalidation when dependencies change
    4. Perfect type inference (since it's used within @property)
    
    Usage:
        @property
        def HSF(self) -> SolarField:
            return component(SolarField(
                zf=constraint(self.zf),
                file_SF=derived(self._file_SF, self.zf)
            ))
    
    Args:
        instance: The component instance to cache and return
        component_name: Optional component name (auto-detected from context)
        
    Returns:
        The same instance, but now tracked for smart caching
        
    Examples:
        @property
        def my_component(self) -> SomeComponent:
            return component(SomeComponent(
                param1=constraint(self.param1),
                param2=constraint(self.param2)
            ))
    """
    
    # Get the current component context (which property is being accessed)
    context = _get_current_component_context()
    
    if context is None:
        # If no context, just return the instance (no caching available)
        return instance
    
    plant_instance, detected_name = context
    final_name = component_name or detected_name
    
    # Check if we should use cached version
    if _should_use_cached_component(plant_instance, final_name):
        cached = _get_cached_component(plant_instance, final_name)
        if cached is not None:
            return cached
    
    # Cache this new instance and return it
    _cache_component(plant_instance, final_name, instance)
    
    return instance

# ============================================================================
# CONSTRAINT AND DERIVED FUNCTIONS FOR @component DECORATOR
# ============================================================================

# Thread-local storage for tracking current component context
from threading import local
_component_context = local()

def _get_current_component_context() -> tuple[object, str] | None:
    """Get the current component being built (plant instance, component name)."""
    
    # Then try to get from calling frame (for component() function)
    try:
        frame = inspect.currentframe()
        # Walk up the call stack to find the property access
        current_frame = frame
        for i in range(5):  # Limit search depth
            if current_frame is None:
                break
            
            frame_locals = current_frame.f_locals
            if 'self' in frame_locals:
                plant_instance = frame_locals['self']
                if hasattr(plant_instance, '_current_property_context'):
                    component_name = getattr(plant_instance, '_current_property_context', None)
                    if component_name:
                        return (plant_instance, component_name)
            
            current_frame = current_frame.f_back
            
    except Exception:
        pass  # Fallback gracefully if frame inspection fails
    
    return None

def _should_use_cached_component(plant_instance, component_name: str) -> bool:
    """Check if cached component is still valid."""
    if not hasattr(plant_instance, '_component_cache'):
        return False
        
    if component_name not in plant_instance._component_cache:
        return False
    
    # Check if dependencies changed (using existing hash system)
    if hasattr(plant_instance, '_component_dependencies'):
        dependencies = plant_instance._component_dependencies.get(component_name, set())
        if dependencies:
            current_hash = plant_instance._compute_params_hash(dependencies)
            cached_hash = plant_instance._param_hash_cache.get(component_name)
            return current_hash == cached_hash
    
    # If no dependencies tracked yet, assume valid (will be tracked on first access)
    return True

def _get_cached_component(plant_instance, component_name: str):
    """Get cached component if available."""
    if hasattr(plant_instance, '_component_cache'):
        return plant_instance._component_cache.get(component_name)
    return None

def _cache_component(plant_instance, component_name: str, instance):
    """Cache a component instance."""
    if not hasattr(plant_instance, '_component_cache'):
        plant_instance._component_cache = {}
    if not hasattr(plant_instance, '_param_hash_cache'):
        plant_instance._param_hash_cache = {}
    
    plant_instance._component_cache[component_name] = instance
    
    # Update hash if dependencies are known
    if hasattr(plant_instance, '_component_dependencies'):
        dependencies = plant_instance._component_dependencies.get(component_name, set())
        if dependencies:
            current_hash = plant_instance._compute_params_hash(dependencies)
            plant_instance._param_hash_cache[component_name] = current_hash

def _set_component_context(plant_instance: object, component_name: str):
    """Set the current component context for dependency tracking."""
    _component_context.current = (plant_instance, component_name)

def _clear_component_context():
    """Clear the current component context."""
    _component_context.current = None

def _register_dependency(param_name: str):
    """Register a dependency for the current component being built."""
    context = _get_current_component_context()
    
    if context is None:
        # If no component context, this is being called outside component creation
        # This is OK - dependency tracking is optional
        return
    
    plant_instance, component_name = context
    
    # Initialize dependencies if not present
    if not hasattr(plant_instance, '_component_dependencies'):
        setattr(plant_instance, '_component_dependencies', {})
    
    dependencies = getattr(plant_instance, '_component_dependencies')
    if component_name not in dependencies:
        dependencies[component_name] = set()
    
    # Add this parameter as a dependency
    dependencies[component_name].add(param_name)

def constraint(value: T, param_name: str | None = None) -> T:
    """
    Mark a parameter as a direct dependency and return it unchanged.
    
    This function serves two purposes:
    1. Returns the input value unchanged (pass-through)
    2. Registers the parameter as a dependency for smart caching
    
    Args:
        value: The parameter value to mark as a constraint
        param_name: Optional parameter name (auto-detected if not provided)
        
    Returns:
        The input value unchanged
        
    Example:
        @component
        def HSF(self) -> SolarField:
            zf = constraint(self.zf)  # Mark zf as dependency
            return SolarField(zf=zf)
    """
    
    # Auto-detect parameter name if not provided
    if param_name is None:
        try:
            # Get the calling frame to extract variable name
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                frame = frame.f_back
            
                # Look for pattern like "self.param_name" in the calling code
                import re
                frame_info = inspect.getframeinfo(frame)
                if frame_info.code_context and len(frame_info.code_context) > 0:
                    source_line = frame_info.code_context[0]
                
                    # Try to extract parameter name from "constraint(self.param_name)"
                    match = re.search(r'constraint\(self\.([a-zA-Z_][a-zA-Z0-9_]*)', source_line)
                    if match:
                        param_name = match.group(1)
                    else:
                        # Fallback: try to extract from assignment like "zf = constraint(self.zf)"
                        match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*constraint', source_line)
                        if match:
                            param_name = match.group(1)
                    
        except:
            pass  # If auto-detection fails, no dependency tracking
    
    # Register dependency if we have a parameter name
    if param_name:
        _register_dependency(param_name)
    
    return value

def derived(callable_func: Callable, *tracked_vars: Any, param_name: str | None = None) -> Any:
    """
    Execute a callable with tracked variables and register dependencies.
    
    This function:
    1. Executes the callable with the tracked variables as arguments
    2. Registers dependencies on the tracked variables for smart caching
    3. Returns the computed result
    
    Args:
        callable_func: Function to execute for computing the derived value
        *tracked_vars: Variables that this derived value depends on
        param_name: Optional parameter name (auto-detected if not provided)
        
    Returns:
        Result of calling callable_func(*tracked_vars)
        
    Example:
        @component
        def HSF(self) -> SolarField:
            def _file_SF(zf):
                return f'dataset_{zf.gv("m"):.0f}m.csv'
            
            zf = constraint(self.zf)
            file_SF = derived(_file_SF, zf)  # Depends on zf
            return SolarField(zf=zf, file_SF=file_SF)
    """
    
    # Auto-detect parameter name if not provided
    if param_name is None:
        try:
            # Get the calling frame to extract variable name
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                frame = frame.f_back
            
                # Look for assignment pattern like "file_SF = derived(...)"
                import re
                frame_info = inspect.getframeinfo(frame)
                if frame_info.code_context and len(frame_info.code_context) > 0:
                    source_line = frame_info.code_context[0]
            
                    match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*derived', source_line)
                    if match:
                        param_name = match.group(1)
                        
        except:
            pass  # If auto-detection fails, no dependency tracking
    
    # Register this as a derived parameter dependency
    if param_name:
        _register_dependency(f"derived_{param_name}")
    
    # For each tracked variable, try to register its parameter name as a dependency
    context = _get_current_component_context()
    if context is not None:
        plant_instance, component_name = context
        
        # Use source code inspection to find parameter names in constraint() calls
        # This avoids the expensive dir() loop that triggers all component properties
        try:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                caller_frame = frame.f_back
                frame_info = inspect.getframeinfo(caller_frame)
                
                if frame_info.code_context:
                    # Look through multiple lines if available (for multi-line derived calls)
                    source_lines = frame_info.code_context
                    source_text = ''.join(source_lines)
                    
                    # Extract constraint parameter names using regex
                    import re
                    # Pattern matches: constraint(self.param_name) 
                    constraint_matches = re.findall(r'constraint\(self\.([a-zA-Z_][a-zA-Z0-9_]*)', source_text)
                    
                    # Register each found parameter name as a dependency
                    for param_name in constraint_matches:
                        if param_name:  # Ensure param_name is not None or empty
                            _register_dependency(param_name)
                        
        except Exception:
            # If source inspection fails, silently continue without dependency tracking
            # This is better than the expensive dir() loop
            pass
    
    # Execute the callable and return result
    try:
        return callable_func(*tracked_vars)
    except Exception as e:
        raise ValueError(f"Error executing derived parameter calculation: {e}")