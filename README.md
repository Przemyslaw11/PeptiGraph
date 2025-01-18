# PeptiGraph: Molecular Fingerprints for Peptide Classification

PeptiGraph is the source code repository for the Advanced Machine Learning course project (2024) at AGH University of Krakow. This project explores peptide classification using molecular fingerprints and compares their performance against ProtBERT embeddings.

## Overview

Peptides are small proteins that play critical roles in living organisms. Like larger proteins, they often serve multiple functions that influence their higher-level properties. Due to their relatively small size, peptides are computationally efficient to process compared to larger proteins.

Traditionally, peptides have not been extensively analyzed as molecular graphs. Instead, sequence-based algorithms dominate peptide analysis. This project aims to evaluate whether molecular fingerprints—a low-level, detailed graph representation—can effectively classify peptides and how this approach compares to established methods such as ProtBERT.

## Project Goals

1. **Dataset Collection**: Gather and process diverse peptide datasets for classification tasks
2. **Fingerprint Analysis**: Evaluate molecular fingerprints using the `scikit-fingerprints` library
3. **Performance Comparison**: Benchmark molecular fingerprints against ProtBERT embeddings

## Datasets

### 1. HemoPI Datasets (1-3)
- Source: [HemoPI Database](https://webs.iiitd.edu.in/raghava/hemopi/datasets.php)
- Publication: [Nature Scientific Reports](https://www.nature.com/articles/srep22843)
- ProtBERT Performance:
  - HemoPI-1: AUROC: 0.975, AUPRC: 0.978
  - HemoPI-2: AUROC: 0.831, AUPRC: 0.811
  - HemoPI-3: AUROC: 0.862, AUPRC: 0.853
 
### 2. Bioactive Peptides Dataset
- Notable for fingerprints outperforming ProtBERT
- Best fingerprint combination (ECFP + TT + AP): AUROC: 90.41%, AUPRC: 95.23%
- ProtBERT Performance: AUROC: 0.755, AUPRC: 0.743

### 3. Grampa Dataset
- ProtBERT Performance: AUROC: 0.863, AUPRC: 0.866
- Best fingerprint combination: ECFP + MACCS + AP + Layered (AUROC: 75.58%, AUPRC: 80.69%)

### 4. Versa Dataset
- Perfect performance achieved with ProtBERT (AUROC: 1.000, AUPRC: 1.000)
- Best fingerprint performance: Topological Torsion (TT) with AUROC: 80.12%, AUPRC: 64.00%

## Fingerprint Configurations

### Single Fingerprints
- ECFP (Extended Connectivity Fingerprints)
- MACCS (Molecular ACCess System)
- TT (Topological Torsion)
- AP (Atom Pairs)
- Layered

### Combinations
- Two Fingerprint Combinations
- Three Fingerprint Combinations
- Four Fingerprint Combinations

## Key Findings

### Molecular Fingerprints Performance
- TT fingerprint consistently outperformed other single configurations
- AP fingerprint showed strong performance in HemoPI and Bioactive Peptides datasets
- Combining multiple fingerprints often improved performance
- Best combinations: TT + Layered, ECFP + TT + AP

### ProtBERT vs Fingerprints
- ProtBERT excelled in sequence-level analysis
- Fingerprints captured crucial low-level structural details
- Dataset characteristics significantly impacted relative performance
- Combined approaches showed promise for future research

## Local Development Setup

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

3. Sync dependencies with `uv`:
    ```bash
    uv sync
    ```

4. Install `pre-commit` hooks:
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Code Quality and Development Tools

### Linter and Formatter

Install Ruff for linting and formatting:

```bash
pip install ruff
```

### Supported Ruff Functions
- **Code Quality Checks**: Pycodestyle, Pyflakes, McCabe, and more
- **Code Formatting**: Ensures consistent line length, quote styles, and trailing commas
- **Imports Management**: Organize and check imports (isort, flake8-tidy-imports)
- **Type Annotations**: Validate annotations (flake8-annotations)
- **Security**: Identify security issues (flake8-bandit)
- **Performance**: Catch potential inefficiencies (flake8-bugbear, flake8-comprehensions)
- **Style**: Enforce style rules (flake8-quotes, pydocstyle)

### VS Code Integration

Create `.vscode/settings.json`:

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

## Common Commands

### Linting and Formatting
```bash
# Check code quality
ruff check .

# Format code
ruff format .

# Lint and fix
ruff check --fix .
```

## References
- [UV Package Manager Documentation](https://github.com/astral-sh/uv)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [scikit-fingerprints package](https://github.com/scikit-fingerprints/scikit-fingerprints)
- [HemoPI Database](https://webs.iiitd.edu.in/raghava/hemopi/datasets.php)
- [Long Range Graph Benchmark](https://arxiv.org/abs/2206.08164)
- [HemoPI Nature Publication](https://www.nature.com/articles/srep22843)
