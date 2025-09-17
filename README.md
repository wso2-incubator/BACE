# APR - Automated Program Repair

A multi-agent system for automated program repair and code generation using large language models.

## Installation

### From source (development mode)

```bash
git clone https://github.com/kaushithamsilva/APR.git
cd APR

# Install the package with human-eval dependency
pip install -e .

# Note: The human-eval package should be cloned from github and installed 
# pip install -e ./human-eval/
```

### Human-Eval Dependency

This project depends on the [human-eval](https://github.com/openai/human-eval) package for code evaluation. You have two options:

1. **Automatic installation** (recommended): The human-eval package will be installed automatically from GitHub when you install this package.

2. **Local installation**: If you have a local copy of human-eval in your project:

   ```bash
   pip install -e ./human-eval/
   ```
