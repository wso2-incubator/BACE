import pytest
from infrastructure.sandbox.types import SandboxConfig
from infrastructure.sandbox.adapters.python import PythonSandbox

def test_python_sandbox_memory_limit_enforcement():
    """
    Test that the PythonSandbox correctly terminates an execution 
    if it exceeds the max_memory_mb limit.
    """
    # Set a very low memory limit for the test so it triggers quickly
    config = SandboxConfig(
        timeout=10,
        max_memory_mb=50,  # 50 MB
    )
    sandbox = PythonSandbox(config)

    # Payload that continuously allocates memory
    memory_hog_code = """
import time
print("Starting memory allocation...")
a = []
for i in range(100):
    a.append(' ' * (10 * 1024 * 1024))  # 10 MB per iteration
    time.sleep(0.05)
print("Done allocation")
"""

    result = sandbox.execute_code(memory_hog_code)
    
    # Assert that execution was terminated
    assert result.success is False
    assert result.error.startswith("Memory limit exceeded")
    
def test_python_sandbox_memory_limit_test_script():
    """
    Test that the PythonSandbox correctly terminates a pytest script execution
    if it exceeds the max_memory_mb limit.
    """
    config = SandboxConfig(
        timeout=10,
        max_memory_mb=50,
    )
    sandbox = PythonSandbox(config)

    # Pytest payload
    memory_hog_test = """
import time

def test_memory_hog():
    a = []
    for i in range(100):
        a.append(' ' * (10 * 1024 * 1024))
        time.sleep(0.05)
"""

    result = sandbox.execute_test_script(memory_hog_test)
    
    assert result.status == "error"
    assert "exceeded memory limit" in (result.error_log or "")

