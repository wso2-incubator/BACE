# How to Add a New Language Support

This guide outlines the steps required to add a new programming language to the coevolution framework. The framework uses a **Facade Pattern** to manage language-specific operations, enforcing a clean Separation of Concerns (SoC).

## 1. Create the Language Directory

Create a new directory under `src/infrastructure/languages/`.
Example for a language named `mojo`:

```bash
mkdir src/infrastructure/languages/mojo
touch src/infrastructure/languages/mojo/__init__.py
```

## 2. Implement the Specialized Protocols

The `ILanguage` interface is a facade that delegates to four specialized protocols defined in `src/coevolution/core/interfaces/language.py`. Each implementation should reside in its own file within the language directory. Any common logic or reusable snippets should be placed in a `helpers.py` or `utils.py` module within the language directory.

### A. ICodeParser (parser.py)

Responsible for static analysis, extracting code from LLM responses, and structural metadata.

```python
# src/infrastructure/languages/mojo/parser.py
from coevolution.core.interfaces.language import ICodeParser

class MojoParser(ICodeParser):
    def extract_code_blocks(self, response: str) -> list[str]: ...
    def is_syntax_valid(self, code: str) -> bool: ...
    def get_structural_metadata(self, code: str) -> dict[str, Any]: ...
    # ... other methods
```

### B. IScriptComposer (script_composer.py)

Responsible for generating executable scripts by combining code snippets with tests or drivers.

```python
# src/infrastructure/languages/mojo/codegen.py
from coevolution.core.interfaces.language import IScriptComposer

class MojoComposer(IScriptComposer):
    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str: ...
    def generate_test_case(self, input_str, output_str, starter_code, test_number) -> str: ...
```

### C. ITestAnalyzer (analyzer.py)

Parses the output of the test command (stdout, stderr, or XML files) into a structured `EvaluationResult`.

```python
# src/infrastructure/languages/mojo/analyzer.py
from coevolution.core.interfaces.language import ITestAnalyzer

class MojoTestAnalyzer(ITestAnalyzer):
    def analyze(self, raw_result: BasicExecutionResult, **kwargs) -> EvaluationResult:
        ...
```

### D. ILanguageRuntime (runtime.py)

Handles language-specific execution commands and environment settings.

```python
class MojoRuntime(ILanguageRuntime):
    @property
    def file_extension(self) -> str:
        return ".mojo"

    def get_execution_command(self, file_path: str) -> list[str]:
        return ["mojo", file_path]

    def get_test_command(self, test_file_path: str, result_xml_path: str, **kwargs) -> list[str]:
        return ["mojo", "test", test_file_path, "--xml", result_xml_path]
```

## 3. Implement the Language Facade

In `src/infrastructure/languages/<lang>/adapter.py`, implement the `ILanguage` protocol. This class aggregates the specialized components by importing them from their respective files.

```python
# src/infrastructure/languages/mojo/facade.py
from coevolution.core.interfaces.language import ILanguage, ICodeParser, IScriptComposer, ILanguageRuntime, ITestAnalyzer
from .parser import MojoParser
from .script_composer import MojoComposer
from .analyzer import MojoTestAnalyzer

class MojoLanguage(ILanguage):
    def __init__(self):
        self._parser = MojoParser()
        self._composer = MojoComposer()
        self._runtime = MojoRuntime() # Implemented in the same file or runtime.py
        self._analyzer = MojoTestAnalyzer()

    @property
    def language(self) -> str:
        return "mojo"

    @property
    def parser(self) -> ICodeParser:
        return self._parser

    @property
    def composer(self) -> IScriptComposer:
        return self._composer

    @property
    def runtime(self) -> ILanguageRuntime:
        return self._runtime

    @property
    def analyzer(self) -> ITestAnalyzer:
        return self._analyzer
```

## 4. Register the Language

Update `src/infrastructure/languages/__init__.py` to include the new language in the factory:

```python
# src/infrastructure/languages/__init__.py
from .mojo.adapter import MojoLanguage
from .python.adapter import PythonLanguage
from .ballerina.adapter import BallerinaLanguage

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

## 5. Verification

Implement unit tests in `tests/test_infrastructure/languages/<lang>/` and verify using:

```bash
uv run mypy src
uv run pytest tests/test_infrastructure/languages/<lang>
```
