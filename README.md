# Roof Analyzer

A toolkit for roof structure analysis, visualization, and solar panel placement optimization.

## Features

- Roof structure modeling
- Roof visualization and drawing tools
- Roof structural expansion algorithm
- Automatic solar panel placement optimization
- Graph Matching Network (GMN) for top view estimation

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Basic Installation

Clone the repository:

```bash
git clone git@github.com:takaaki5564/roof-analyzer.git
cd roof-analyzer
```

### CPU Environment Installation

For machines without a GPU:

```bash
# Option 1: Using requirements file
pip install -r requirements-cpu.txt

# Option 2: Using setup.py with CPU extras
pip install -e ".[cpu]"
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```

### GPU Environment Installation (CUDA 11.5)

For machines with NVIDIA GPU and CUDA 11.5:

```bash
# Option 1: Using requirements file
pip install -r requirements-cuda11.5.txt

# Option 2: Using setup.py with GPU extras
pip install -e ".[gpu]"
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu115.html
```

