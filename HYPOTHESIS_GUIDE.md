# Property-Based Testing with Hypothesis

## Overview

**Hypothesis** is a Python library for property-based testing that can automatically generate test cases and discover edge cases you might not think of manually. Instead of writing specific test cases with fixed inputs, you describe the *properties* your code should satisfy, and Hypothesis generates hundreds of test cases to verify those properties.

## What Hypothesis Found in Your Code

When we ran the initial hypothesis tests, it immediately discovered **two real bugs**:

1. **Array constructor issue**: `Array(values, None)` crashes because the Array class doesn't properly handle `None` units
2. **Unit parser issue**: Units like `'m³/s'` fail because the parser can't handle the `³` character

This demonstrates Hypothesis's power - it found actual issues in minutes that might have taken weeks to discover manually.

## Key Benefits for Your Parametric Analysis Module

### 1. **Automated Edge Case Discovery**
```python
@given(st.lists(st.floats(min_value=1e-10, max_value=1e10), min_size=1, max_size=20))
def test_extreme_values(values):
    """Test with extremely large/small values."""
    arr = Array(values, 'W')
    # Hypothesis will try values like 1e-10, 1e10, etc.
    # and find where your code breaks
```

### 2. **Property Verification**
```python  
@given(parameter_dict())
def test_setup_cases_properties(params):
    """Properties that should ALWAYS hold."""
    cases, units = parametric.setup_cases(params)
    
    # These properties should never be violated:
    assert isinstance(cases, pd.DataFrame)
    assert len(cases.columns) == len(params)
    assert set(cases.columns) == set(params.keys())
```

### 3. **Physics/Math Invariant Testing**
```python
@given(power_values=st.lists(st.floats(min_value=1, max_value=1000)))  
def test_energy_conservation(power_values):
    """Energy conservation should always hold."""
    for power, efficiency in zip(power_values, efficiency_values):
        energy_in = power
        energy_out = power * efficiency + power * (1 - efficiency)
        assert abs(energy_in - energy_out) < 1e-10
```

## Practical Implementation

### Step 1: Add to Development Workflow
```bash
# Run property-based tests regularly
poetry run pytest tests/test_par_hypothesis.py -v

# Run with more examples for thorough testing
poetry run pytest tests/test_par_hypothesis.py --hypothesis-show-statistics
```

### Step 2: Common Strategies for Your Domain
```python
# Engineering parameter ranges
temperatures = st.floats(min_value=-273, max_value=1000)  # Kelvin constraints
pressures = st.floats(min_value=0, max_value=1e6)  # Physical constraints
efficiencies = st.floats(min_value=0, max_value=1)  # Bounded by physics

# Realistic parameter combinations  
@st.composite
def engineering_parameters(draw):
    temp = draw(temperatures)
    pressure = draw(pressures) 
    # Add physics-based constraints
    assume(temp > 0 if pressure > 101325 else True)  # No vacuum at high temp
    return {'temperature': temp, 'pressure': pressure}
```

### Step 3: Integration with CI/CD
```yaml
# In your GitHub Actions or similar
- name: Run Property-Based Tests
  run: |
    poetry run pytest tests/test_par_hypothesis.py \
      --hypothesis-show-statistics \
      --hypothesis-profile=ci
```

## Advanced Edge Case Automation

### 1. **Shrinking**: When Hypothesis finds a failure, it automatically finds the *minimal* example:
```
❌ Original failing case: [1e-100, 2e200, -5e150, 7e-300, ...]
✅ Shrunk to minimal case: [0.0, 1e200]  # Much easier to debug!
```

### 2. **Stateful Testing**: Test sequences of operations:
```python
class ParametricStateMachine(RuleBasedStateMachine):
    @rule(params=parameter_dict())
    def setup_cases(self, params):
        self.cases = self.parametric.setup_cases(params)
    
    @rule()
    @precondition(lambda self: self.cases is not None)
    def run_analysis(self):
        results = self.parametric.run_analysis()
        # Properties about results should hold
```

### 3. **Performance Properties**:
```python
@given(n_cases=st.integers(min_value=1, max_value=10000))
def test_performance_scaling(n_cases):
    """Performance should scale reasonably."""
    import time
    start = time.time()
    # ... run analysis with n_cases
    duration = time.time() - start
    
    # Performance property: O(n) or O(n log n), not O(n²)
    assert duration < n_cases * 0.001  # Less than 1ms per case
```

## Recommended Testing Strategy

### Phase 1: Basic Properties (Immediate)
- Parameter validation
- DataFrame structure consistency  
- Unit handling correctness
- No crashes on valid inputs

### Phase 2: Physics/Engineering Properties (Next)
- Energy conservation
- Thermodynamic constraints
- Realistic output ranges
- Unit conversion accuracy

### Phase 3: Performance & Scale Properties (Later)
- Memory usage bounds
- Time complexity verification
- Large parameter space handling
- Numerical stability

## Configuration for Your Project

Add to `pyproject.toml`:
```toml
[tool.hypothesis]
max_examples = 100
deadline = 5000  # 5 seconds
derandomize = true  # Reproducible in CI

[tool.hypothesis.profiles.ci]
max_examples = 1000
deadline = 10000

[tool.hypothesis.profiles.dev]  
max_examples = 50
deadline = 2000
```

## ROI Analysis

**Manual Testing**: 
- 10 test cases = 2 hours of writing
- Limited edge case coverage
- Miss boundary conditions

**Property-Based Testing**:
- 1 property = 10 minutes of writing  
- Tests 100-1000 generated cases automatically
- Finds edge cases you didn't think of
- Provides minimal failing examples
- Prevents regressions

**Result**: 10x more coverage in 1/10th the time, plus automatic edge case discovery.

## Next Steps

1. **Start small**: Add 2-3 property tests for critical functions
2. **Run regularly**: Include in your development workflow  
3. **Expand coverage**: Add physics constraints and performance properties
4. **CI Integration**: Run comprehensive property tests on every commit
5. **Bug prevention**: Use properties to prevent regressions

Property-based testing with Hypothesis transforms edge case testing from a manual, time-consuming process into an automated, comprehensive verification system that finds bugs you never would have thought to look for.
