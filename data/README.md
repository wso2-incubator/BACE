# Data Directory Structure

This directory contains all datasets, generated code completions, evaluation results, and logs for the APR (Automated Program Repair) research project.

## 📁 Directory Overview

```
data/
└── human_eval/
    ├── HumanEval_20.jsonl.gz          # 20-problem subset for development
    ├── evaluations/                   # Evaluation reports and analyses
    ├── generations/                   # Generated code completions
    │   ├── agent_coder/              # Multi-agent AgentCoder results
    │   ├── simple/                   # Single-agent baseline results
    │   └── *.jsonl                   # Individual model results
    └── logs/                         # Execution logs and debug information
```

## 📊 Datasets

### HumanEval_20.jsonl.gz

- **Purpose**: Reproducible 20-problem subset of HumanEval for faster development and testing
- **Creation**: Generated using `experiments/scripts/generate_humaneval_subset.py` with seed=42
- **Problems**: Randomly selected from the full HumanEval dataset
- **Usage**: Default dataset for subset-based experiments

### Full HumanEval Dataset

- **Location**: Accessed via `human-eval/` submodule
- **Problems**: Complete 164-problem HumanEval benchmark
- **Usage**: Production experiments and final evaluations

## 🤖 Generated Completions (`generations/`)

### File Naming Convention

All generated files follow a consistent naming pattern:

```
{model_name}-{dataset_info}-{parameters}-{suffix}.jsonl
```

**Examples:**

- `qwen2.5-coder_7b-subset20-max3-numSamples1.jsonl`
- `qwen2.5-coder_7b-samples5-detailed.jsonl`

### Directory Structure

#### `agent_coder/`

**Multi-agent AgentCoder results with iterative improvement**

- **Final Results**: `{model}-{dataset}-{params}.jsonl`
- **Iteration Files**: `{model}-{dataset}-{params}-iter{N}.jsonl`
- **Evaluation Results**: `{filename}_results.jsonl`

**Key Features:**

- Multi-agent workflow (programmer → tester → executor)
- Iterative code improvement based on test feedback
- Separate files for each iteration to track progress
- Professional error handling and recovery

**Example Files:**

```
qwen2.5-coder_7b-subset20-max3-numSamples1.jsonl              # Final results
qwen2.5-coder_7b-subset20-max3-numSamples1-iter1.jsonl        # First iteration
qwen2.5-coder_7b-subset20-max3-numSamples1-iter2.jsonl        # Second iteration
qwen2.5-coder_7b-subset20-max3-numSamples1-iter3.jsonl        # Third iteration
```

#### `simple/`

**Single-agent baseline results**

- **Purpose**: Baseline comparisons to multi-agent approach
- **Method**: Direct LLM prompting without iterative improvement
- **Customizable**: Configurable prompt templates and parameters

**Example Files:**

```
qwen2.5-coder_7b_subset20-samples1-basic.jsonl               # Basic prompt
qwen2.5-coder_7b_subset20-samples1-detailed.jsonl            # Detailed prompt
qwen2.5-coder_7b_subset20-samples1--no-prompt.jsonl          # Custom experiment
```

#### Individual Model Files

Direct results from various models and configurations:

```
qwen2.5-coder:7b.jsonl          # Standard 7B model
qwen2.5-coder:14b.jsonl         # Larger 14B model  
gpt-oss:20b.jsonl               # Alternative model
```

## 📈 Evaluation Results

### Result Files

- **Naming**: `{generation_file}_results.jsonl`
- **Content**: Pass@k metrics, execution results, statistical analysis
- **Evaluation**: Generated using HumanEval evaluation framework

### Metrics Included

- **pass@1**: Percentage of problems solved on first try
- **pass@5**: Percentage of problems solved in 5 attempts
- **pass@10**: Percentage of problems solved in 10 attempts
- **Execution Details**: Success/failure breakdown per problem

## 📝 Logs Directory (`logs/`)

### Content

- Execution traces from AgentCoder runs
- Debug information and error logs
- Performance metrics and timing data
- Agent communication logs

### Log Files

- **Format**: Timestamped entries with structured data
- **Rotation**: Organized by experiment runs
- **Debug**: Detailed information for troubleshooting

## 🔧 Usage Examples

### Generate HumanEval Subset

```bash
cd /Users/kaushitha/Documents/APR
./APR_env/bin/python experiments/scripts/generate_humaneval_subset.py
```

### Run AgentCoder Experiment

```bash
./APR_env/bin/python experiments/scripts/agent_coder_generator.py
```

### Run Baseline Experiment  

```bash
./APR_env/bin/python experiments/scripts/simple_generator.py
```

### Evaluate Results

```bash
./APR_env/bin/python experiments/scripts/evaluate_humaneval_subset.py sample_file.jsonl
```

## 📋 File Format Specifications

### Generation Files (`.jsonl`)

Each line contains a JSON object with:

```json
{
    "task_id": "HumanEval/0",
    "completion": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    # implementation"
}
```

### Result Files (`_results.jsonl`)

Contains evaluation metrics and detailed results:

```json
{
    "pass@1": 0.85,
    "pass@5": 0.92,
    "pass@10": 0.95,
    "details": {...}
}
```

## 🎯 Best Practices

### File Organization

1. **Use descriptive filenames** that include model, dataset, and parameters
2. **Keep results with their source files** using `_results.jsonl` suffix
3. **Organize by approach** (agent_coder vs simple vs individual models)
4. **Archive old experiments** to maintain clean working directories

### Experiment Workflow

1. **Start with subset** (`HumanEval_20.jsonl.gz`) for rapid iteration
2. **Use consistent naming** for easy comparison across experiments
3. **Save iteration files** to track improvement progress
4. **Evaluate immediately** after generation for quick feedback

### Data Management

- **Version Control**: Track dataset versions and generation parameters
- **Backup Important Results**: Copy significant results to safe storage
- **Clean Regularly**: Remove temporary and debug files periodically
- **Document Experiments**: Use meaningful suffixes and maintain experiment logs

## 🔍 Analysis and Comparison

### Performance Comparison

Compare different approaches using the evaluation results:

```bash
# Compare AgentCoder vs Simple baseline
ls data/human_eval/generations/agent_coder/*_results.jsonl
ls data/human_eval/generations/simple/*_results.jsonl
```

### Iteration Analysis

Track improvement across AgentCoder iterations:

```bash
ls data/human_eval/generations/agent_coder/*iter*_results.jsonl
```

### Model Comparison

Evaluate different model sizes and types:

```bash
ls data/human_eval/generations/*7b*_results.jsonl
ls data/human_eval/generations/*14b*_results.jsonl
```

---

## 📞 Support

For questions about data formats, evaluation procedures, or experiment setup:

- Check experiment scripts in `experiments/scripts/`
- Review configuration options in script documentation
- Examine existing result files for format examples

**Generated**: 2025-09-26  
**Project**: APR Research - Automated Program Repair using LLMs
