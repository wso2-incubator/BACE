# APR Coevolution Configuration System

This directory contains modular YAML configuration files for coevolution experiments.

## Structure

```
configs/
├── llm/                    # LLM provider and model configs
├── profiles/
│   ├── code/               # Code population profiles
│   ├── unittest/           # Unit test population profiles
│   ├── differential/       # Differential test profiles
│   └── public/            # Public test profiles
├── sandbox/               # Sandbox and worker configs
├── schedules/             # Evolution schedules
└── experiments/           # Complete experiment configs
```

## Usage

### Run with default config

```bash
uv run python main.py run --config configs/experiments/default.yaml
```

### Override specific components

```bash
# Use different LLM
uv run python main.py run --config configs/experiments/default.yaml \
  --llm configs/llm/gpt-4.yaml

# Override dataset parameters
uv run python main.py run --config configs/experiments/default.yaml \
  --difficulty easy \
  --start-index 0 --end-index 10
```

### Quick test

```bash
uv run python main.py run --config configs/experiments/quick-test.yaml
```

### List available configs

```bash
uv run python main.py list-configs llm
uv run python main.py list-configs code
uv run python main.py list-configs experiment
```

### Dry run (validate config without running)

```bash
uv run python main.py run --config configs/experiments/default.yaml --dry-run
```

## Creating New Configs

### 1. Create a new LLM config

```yaml
# configs/llm/my-model.yaml
provider: "openai"
model: "my-model-name"
reasoning_effort: "medium"
temperature: 0.7
```

### 2. Create a new code profile

```yaml
# configs/profiles/code/my-profile.yaml
initial_population_size: 10
max_population_size: 15
mutation_rate: 0.3
crossover_rate: 0.2
edit_rate: 0.5
k_failing_tests: 10
# ... other parameters
```

### 3. Create a new experiment

```yaml
# configs/experiments/my-experiment.yaml
experiment:
  name: "my-experiment"
  description: "My custom experiment"

# Reference existing configs
llm: "llm/gpt-4.yaml"
code_profile: "profiles/code/aggressive.yaml"
unittest_profile: "profiles/unittest/standard.yaml"
differential_profile: "profiles/differential/standard.yaml"
public_profile: "profiles/public/standard.yaml"
sandbox: "sandbox/sandbox-default.yaml"
schedule: "schedules/alternating-6gen.yaml"

# Dataset configuration
dataset:
  version: "release_v6"
  start_date: "2025-03-01"
  end_date: "2025-05-10"
  difficulty: "hard"

subset:
  start_index: 0
  end_index: 100

logging:
  console_level: "INFO"
  file_level: "DEBUG"
```

## Environment Variables

Configs support environment variable substitution:

```yaml
# Use ${VAR_NAME} syntax
api_key: "${OPENAI_API_KEY}"

# With default value
model: "${MODEL_NAME:-gpt-5-mini}"
```

## Configuration Precedence

1. **Base config** (if exists): `configs/base.yaml`
2. **Experiment config**: The file you specify
3. **CLI overrides**: Command-line arguments

Higher precedence values override lower ones.
