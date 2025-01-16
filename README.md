Here's an updated README that includes a concise list of supported Ruff functions:

---

# PeptiGraph: Molecular Fingerprints for Peptide Classification

PeptiGraph is the source code repository for the Advanced Machine Learning course project (2024) at AGH University of Krakow. This project explores peptide classification using molecular fingerprints and compares their performance against ProtBERT embeddings and protein descriptors from PyBioMed.

## Overview

Peptides are small proteins that play critical roles in living organisms. Like larger proteins, they often serve multiple functions that influence their higher-level properties. Due to their relatively small size, peptides are computationally efficient to process compared to larger proteins.

Traditionally, peptides have not been extensively analyzed as molecular graphs. Instead, sequence-based algorithms dominate peptide analysis. This project aims to evaluate whether molecular fingerprints—a low-level, detailed graph representation—can effectively classify peptides and how this approach compares to established methods such as ProtBERT and PyBioMed descriptors.

## Project Goals

1. **Dataset Collection**  
   Gather datasets suitable for peptide classification.
   
2. **Fingerprint Analysis**  
   Evaluate molecular fingerprints (using the `scikit-fingerprints` library) for peptide classification.
   
3. **Performance Comparison**  
   Benchmark molecular fingerprints against ProtBERT embeddings and PyBioMed descriptors.

## Expected Outcomes

- A comprehensive evaluation of molecular fingerprints for peptide classification.
- Comparative insights into the performance of ProtBERT and PyBioMed descriptors.

## Datasets

The following datasets are utilized for classification tasks:

1. **Peptides-struct and Peptides-func**  
   From the Long Range Graph Benchmark ([arXiv link](https://arxiv.org/abs/2206.08164)).
   
2. **HemoPI Datasets (1-3)**  
   Accessible via [HemoPI](https://webs.iiitd.edu.in/raghava/hemopi/datasets.php) and related publications ([Nature article](https://www.nature.com/articles/srep22843)).

---

## Local Development Setup

Follow these steps to set up the project for local development.

### Prerequisites

Install the UV package manager for fast dependency management:

```bash
# Install UV
pip install uv

# Set UV as default pip (add to your shell configuration file For Bash (~/.bashrc) or ZSH (~/.zshrc))
alias pip='uv pip'
```

### Project Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Przemyslaw11/PeptiGraph.git
    cd PeptiGraph
    ```

2. Create and activate a virtual environment:
    ```bash
    # Create virtual environment
    uv venv

    # Activate virtual environment
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

3. Install project dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Code Quality and Development Tools

### Linter and Formatter

Install Ruff for linting and formatting:

```bash
pip install ruff
```

### Supported Ruff Functions
- **Code Quality Checks**: Pycodestyle, Pyflakes, McCabe, and more.
- **Code Formatting**: Ensures consistent line length, quote styles, and trailing commas.
- **Imports Management**: Organize and check imports (isort, flake8-tidy-imports).
- **Type Annotations**: Validate annotations (flake8-annotations).
- **Security**: Identify security issues (flake8-bandit).
- **Performance**: Catch potential inefficiencies (flake8-bugbear, flake8-comprehensions).
- **Style**: Enforce style rules (flake8-quotes, pydocstyle).

For the full list of active rules, refer to the `pyproject.toml` file in this repository.

### VS Code Integration

For seamless development in VS Code, create a `.vscode/settings.json` file with the following configuration:

```json
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": true,
            "source.organizeImports": true
        }
    }
}
```

---

## Common Commands

### Run Linter
Check code quality using Ruff:
```bash
ruff check .
```

### Format Code
Automatically format code:
```bash
ruff format .
```

### Lint and Fix
Lint and fix issues in one command:
```bash
ruff check --fix .
```

---

## References

- **UV Documentation**: [UV Package Manager](https://github.com/astral-sh/uv)  
- **Ruff Documentation**: [Ruff Python Linter](https://docs.astral.sh/ruff/)

--- 