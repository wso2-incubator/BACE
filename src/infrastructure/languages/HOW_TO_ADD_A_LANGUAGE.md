# How to Add a New Language Support

This guide outlines the steps required to add a new programming language to the coevolution framework.

## 1. Create the Language Directory
Create a new directory under `src/infrastructure/languages/`. 
Example for a language named `mojo`:
```bash
mkdir src/infrastructure/languages/mojo
touch src/infrastructure/languages/mojo/__init__.py
```

## 2. Implement the Language Protocol
Create `src/infrastructure/languages/<lang>/adapter.py`. This file must contain a class that implements the `ILanguage` protocol defined in `src/coevolution/core/interfaces/language.py`.

### Recommended Folder Structure
To keep the adapter clean, delegate parsing and code generation to separate modules:
- `adapter.py`: Orchestration and execution configuration.
- `ast.py` or `parser.py`: Syntax analysis, metadata extraction, validation.
- `codegen.py`: Script composition and template generation.
- `analyzer.py`: Implementation of `ITestAnalyzer` to parse test results.

### Key Methods to Implement:
- `language`: Return the string name of the language (e.g., "python").
- `extract_code_blocks`: Extract code from LLM Markdown responses using regex.
- `is_syntax_valid`: Basic or AST-based syntax check.
- `get_execution_command`: CLI command to run a script (e.g., `["python", "script.py"]`).
- `get_test_command`: CLI command to run tests (e.g., `["pytest", "test.py"]`).
- `compose_test_script`: Combine implementation + tests into one executable script.
- `parse_test_results`: Delegate to a local `analyzer.py` that implements `ITestAnalyzer`.
- `get_structural_metadata`: Extract function/class names and imports.

## 3. Implement a Test Analyzer
Create `src/infrastructure/languages/<lang>/analyzer.py`. This class must implement the `ITestAnalyzer` protocol from `src/coevolution/core/interfaces/analyzer.py`.

```python
from coevolution.core.interfaces.language import ITestAnalyzer
from coevolution.core.interfaces.data import BasicExecutionResult, EvaluationResult

class MyLanguageAnalyzer(ITestAnalyzer):
    def analyze(self, raw_result: BasicExecutionResult, **kwargs: Any) -> EvaluationResult:
        # Implementation here...
        ...
```

## 4. Register the Language
Update `src/infrastructure/languages/__init__.py`:
1. Import your new Language class.
2. Update the `create_language_adapter` factory function to include your language.

```python
# src/infrastructure/languages/__init__.py
def create_language_adapter(language: str) -> ILanguage:
    lang_lower = language.lower()
    if lang_lower == "python":
        return PythonLanguage()
    elif lang_lower == "ballerina":
        return BallerinaLanguage()
    elif lang_lower == "mojo":
        return MojoLanguage()
    ...
```

## 5. Add Unit Tests
Create a corresponding test directory: `tests/test_infrastructure/languages/<lang>/`.
Implement unit tests for your adapter, parser, codegen, and analyzer modules.

### Verification
Run mypy and pytest to ensure your implementation strictly follows the protocol:
```bash
uv run mypy src
uv run pytest tests/test_infrastructure/languages/<lang>
```
