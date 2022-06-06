# Baba Is Smart

A model zoo of level-solving agents:

- A\*
- IDA\*
- IDAQ\*
- Soft Actor Critic
- I2A

We also provide utilites for convering between the Keke Competition and the `baba-is-auto` formats.

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
pip install -e .
cd baba-is-auto && pip install -U .
```
