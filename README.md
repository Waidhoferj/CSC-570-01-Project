# Baba Is You Agent

## Getting started

1. Clone the repository
2. Install submodules:

```
git submodule update --init --recursive
```

3. Install dependencies

```
conda env create --file environment.yml
conda activate baba-ai
pre-commit install
cd baba-is-auto && pip install -U .
```
