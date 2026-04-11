# LLM Client Module

A unified interface for working with different Large Language Model (LLM) providers.

## Structure

```markdown
llm_client/
├── __init__.py          # Public API exports
├── base.py              # Abstract base class and core functionality
├── exceptions.py        # Custom exceptions
├── openai.py           # OpenAI implementations (Chat & Response APIs)
├── ollama.py           # Ollama implementation
├── factory.py          # Factory function for creating clients
└── py.typed            # Type hint marker
```

## Features

- **Unified Interface**: Abstract base class (`LLMClient`) for consistent API across providers
- **Token Tracking**: Built-in token counting and limits
- **Multiple Providers**: Support for OpenAI (Chat & Response APIs) and Ollama
- **Type Safety**: Full type hints with `py.typed` marker
- **Flexible Configuration**: Per-client token limits and provider-specific options

## Usage

### Basic Usage

```python
from infrastructure.llm_client import create_llm_client

# Create a client using the factory
client = create_llm_client(
    provider="openai",
    model="gpt-4",
    max_output_tokens=100000,
    enable_token_limit=True
)

# Generate text
response = client.generate("What is the capital of France?")
print(response)

# Check token usage
print(f"Tokens used: {client.total_output_tokens}")
print(f"Tokens remaining: {client.tokens_remaining}")
```

### Using Specific Implementations

```python
from infrastructure.llm_client import OpenAIClient, OllamaClient

# OpenAI with reasoning effort
openai_client = OpenAIClient(
    model="o1",
    reasoning_effort="medium",
    max_output_tokens=500000
)

# Ollama (token limits disabled by default)
ollama_client = OllamaClient(
    model="llama2",
    enable_token_limit=False
)
```

### Token Management

```python
# Reset token count
client.reset_token_count()

# Modify token limits
client.set_token_limit(max_tokens=200000, enabled=True)

# Temporarily disable limits
client.disable_token_limit()

# Re-enable limits
client.enable_token_limit(max_tokens=100000)

# Check status
print(f"Limit enabled: {client.is_token_limit_enabled}")
print(f"Max tokens: {client.max_output_tokens}")
```

## Supported Providers

### OpenAI

**Provider ID**: `"openai"` (Response API) or `"openai-chat"` (Chat API)

**Models**: `gpt-4`, `gpt-3.5-turbo`, `o1`, etc.

**Special Features**:

- Reasoning effort control (Response API only)
- Accurate token tracking from API responses
- Token limits enabled by default

**Example**:

```python
client = create_llm_client(
    provider="openai",
    model="o1",
    reasoning_effort="high",
    max_output_tokens=100000
)
```

### Ollama

**Provider ID**: `"ollama"`

**Models**: `llama2`, `codellama`, `mistral`, etc.

**Special Features**:

- Local model execution
- Token limits disabled by default
- Estimated token counting

**Example**:

```python
client = create_llm_client(
    provider="ollama",
    model="llama2",
    enable_token_limit=False
)
```

## Exception Handling

```python
from infrastructure.llm_client import TokenLimitExceededError

try:
    response = client.generate(prompt)
except TokenLimitExceededError as e:
    print(f"Token limit exceeded!")
    print(f"Current tokens: {e.current_tokens}")
    print(f"Limit: {e.limit}")
```

## Design Principles

1. **Separation of Concerns**: Each provider in its own module
2. **Single Responsibility**: Base class handles common logic, providers handle specifics
3. **Open/Closed**: Easy to add new providers without modifying existing code
4. **Dependency Inversion**: Depend on abstractions (LLMClient), not implementations

## Adding New Providers

To add a new provider:

1. Create a new file (e.g., `anthropic.py`)
2. Implement a class inheriting from `LLMClient`
3. Implement the `generate()` method
4. Add the provider to `factory.py`
5. Export it from `__init__.py`

Example:

```python
# anthropic.py
from .base import LLMClient

class AnthropicClient(LLMClient):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Initialize Anthropic client
    
    def generate(self, prompt, **kwargs):
        # Implement generation logic
        pass
```
