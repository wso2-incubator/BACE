# Sandbox Module

Safe code execution environment for testing generated code with comprehensive test analysis.

## Structure

```markdown
sandbox/
├── __init__.py          # Public API exports
├── types.py             # Data classes (TestResult, TestExecutionResult, etc.)
├── exceptions.py        # Custom exceptions
├── core.py              # SafeCodeSandbox implementation
├── analyzer.py          # PytestXmlAnalyzer for parsing test results
├── executor.py          # TestExecutor for orchestrating execution
├── utils.py             # Factory functions and utilities
└── py.typed             # Type hints marker
```

## Features

- **Safe Execution**: Restricted environment with blocked dangerous operations
- **Test Analysis**: Comprehensive pytest/unittest test result parsing
- **Test Ordering**: Guarantees test results match script order (critical for coevolution)
- **Timeout Control**: Per-test and per-script timeout configuration
- **XML Parsing**: Robust JUnit XML analysis with ANSI code sanitization
- **Type Safety**: Full type hints with comprehensive data classes

## Usage

### High-Level API (Recommended)

```python
from infrastructure.sandbox import create_test_executor, check_test_execution_status

# Create executor with default safe configuration
executor = create_test_executor(test_method_timeout=30)

test_script = """
import unittest

def add(a, b):
    return a + b

class TestAdd(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_negative(self):
        self.assertEqual(add(-1, -1), -2)
"""

# Execute test script
result = executor.execute_test_script(test_script)

# Check results
print(f"Passed: {result.tests_passed}")
print(f"Failed: {result.tests_failed}")
print(f"Status: {check_test_execution_status(result)}")

# Iterate through individual test results (guaranteed in script order)
for test in result.test_results:
    print(f"  {test.name}: {test.status}")
    if test.details:
        print(f"    {test.details}")
```

### Low-Level API (More Control)

```python
from infrastructure.sandbox import SafeCodeSandbox, PytestXmlAnalyzer

# Create sandbox with custom configuration
sandbox = SafeCodeSandbox(
    timeout=60,
    max_memory_mb=200,
    max_output_size=100000,
    test_method_timeout=15,
    allowed_imports=["math", "random", "itertools"]
)

# Execute simple code
code_result = sandbox.execute_code("print('Hello')")
print(code_result.output)

# Execute test script with full analysis
test_result = sandbox.execute_test_script(test_script)
```

### Factory Functions

```python
from infrastructure.sandbox import create_safe_test_environment, create_test_executor

# Create sandbox with defaults
sandbox = create_safe_test_environment(
    test_method_timeout=30,
    script_timeout=300
)

# Create executor with defaults
executor = create_test_executor(test_method_timeout=30)
```

## Data Classes

### EvaluationResult (from core.interfaces)

A unified test case result used throughout the framework:

```python
@dataclass(frozen=True)
class EvaluationResult:
    status: Literal["passed", "failed", "error"]
    error_log: Optional[str] = None
    execution_time: float = 0.0
```

### BasicExecutionResult

Basic code execution result:

```python
@dataclass
class BasicExecutionResult:
    success: bool           # Execution succeeded
    output: str            # stdout content
    error: str             # stderr content
    execution_time: float  # Time in seconds
    timeout: bool          # Whether execution timed out
    return_code: int       # Process return code
```

### SandboxConfig

Serializable configuration for creating sandboxes:

```python
@dataclass(frozen=True)
class SandboxConfig:
    timeout: int = 30
    max_memory_mb: int = 100
    max_output_size: int = 10000
    allowed_imports: Optional[List[str]] = None
    python_executable: Optional[str] = None
    test_method_timeout: Optional[int] = None
```

## Key Features

### Test Result Ordering

**CRITICAL**: The `test_results` list is **always** in the same order as test methods appear in the script, regardless of pytest's execution order. This is essential for:

- Building observation matrices in coevolution
- Generating targeted feedback for specific tests
- Consistent indexing across multiple runs

```python
# Test methods in script:
# 1. test_edge_case
# 2. test_normal_case
# 3. test_invalid_input

result = executor.execute_test_script(test_script)

# result.test_results[0] is ALWAYS test_edge_case
# result.test_results[1] is ALWAYS test_normal_case
# result.test_results[2] is ALWAYS test_invalid_input
```

### Safety Restrictions

The sandbox blocks potentially dangerous operations:

- File I/O (`open`, `file`)
- Network operations (`socket`, `urllib`, `requests`)
- System operations (`os`, `subprocess`, `shutil`)
- Code execution (`exec`, `eval`, `compile`)
- Introspection (`globals`, `locals`, `vars`, `dir`)

### Timeout Configuration

Two-level timeout control:

1. **Per-test timeout**: Individual test method limit (default: 30s)
2. **Script timeout**: Entire test suite limit (default: 180s)

```python
executor = TestExecutor(
    timeout=300,              # 5 minutes for entire script
    test_method_timeout=30    # 30 seconds per test method
)
```

### XML Analysis

Robust pytest JUnit XML parsing with:

- ANSI escape code stripping
- Temporary file path sanitization
- Module prefix cleanup
- Fallback error analysis

## Components

### SafeCodeSandbox

Core sandbox implementation providing:

- Code safety checks
- Process isolation
- Timeout enforcement
- Test script execution
- Result ordering

### TestExecutor

High-level orchestration combining:

- SafeCodeSandbox for execution
- PytestXmlAnalyzer for result parsing
- Automatic test result reordering

### PytestXmlAnalyzer

Parses pytest JUnit XML output:

- Extracts test names, status, and details
- Sanitizes error messages
- Detects script-level errors
- Provides fallback analysis

## Design Principles

1. **Separation of Concerns**: Each component handles one responsibility
2. **Composability**: Components can be used independently or together
3. **Type Safety**: Comprehensive type hints throughout
4. **Robustness**: Graceful error handling and fallback mechanisms
5. **Extensibility**: Easy to add new analyzers or execution modes

## Migration from Old API

The refactored API maintains backward compatibility:

```python
# Old import (still works)
from infrastructure.sandbox import SafeCodeSandbox, TestExecutor

# New organized imports (recommended)
from infrastructure.sandbox import (
    SafeCodeSandbox,
    TestExecutor,
    create_test_executor,
    TestExecutionResult,
)
```

All public APIs remain unchanged, so existing code continues to work without modifications.
