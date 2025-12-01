
# Triton Introduction

Triton is an open-source programming language and compiler developed by OpenAI for writing highly efficient GPU code, especially for machine learning and AI workloads. Triton enables researchers and developers to write custom GPU kernels in Python-like syntax, making GPU programming more accessible and productive. However, writing efficient GPU code can still be challenging, and Triton aims to simplify this process and I'm using CUDA 12.9/13 on Windows as the GPU backend to run Triton.

## Key Features

- Pythonic syntax for easy learning
- Automatic parallelization and optimization
- Seamless integration with PyTorch
- Open-source and actively developed

## Triton Architecture

Below is a simplified architecture diagram of Triton:

```mermaid
graph TD
 A[Python Code] --> B[Triton Compiler]
 B --> C[LLVM IR]
 C --> D[GPU Binary]
 D --> E[GPU Execution]
```

### General Triton Workflow

```mermaid
sequenceDiagram
 participant User
 participant Triton
 participant LLVM
 participant GPU
 User->>Triton: Write kernel in Python
 Triton->>LLVM: Compile to LLVM IR
 LLVM->>GPU: Generate GPU binary
 GPU->>User: Execute kernel and return results
```

### Triton & CUDA Workflow

```mermaid
sequenceDiagram
 participant User
 participant Triton
 participant CUDA
 participant GPU
 User->>Triton: Write kernel in Python
 Triton->>CUDA: Compile to PTX (via LLVM)
 CUDA->>GPU: Load and execute binary
 GPU->>User: Return results
```

## Example Use Case

Triton is ideal for custom deep learning layers, scientific computing, and accelerating AI research. It is used by leading organizations to optimize performance-critical workloads.

---

## Export Packages to requirements.txt using uv export

```bash
uv export --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu130 --format requirements-txt --all-packages --index-strategy unsafe-best-match > requirements.txt
```

## Sync Environment

```bash
uv sync -U --index-strategy unsafe-best-match --all-packages --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu130
```

## Lock and Upgrade Package

```bash
uv lock --upgrade --index-strategy unsafe-best-match --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu130
```
