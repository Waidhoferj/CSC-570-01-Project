# Baba Is Smart

> Course: Cal Poly CSC 570-01 \
> Professor: Rodrigo Canaan \
> Term: Spring 2022

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

## Results

0.65 128.4 4.9
Running our models on converted stages from the Keke Competition with a 70-30 train/test split:
| Model | Win Rate | Avg Reward | Avg Steps |
|--------|----------|-----------|-----------|
| Random |0.33 | -33.2 | 119.7|
| IDA\* |0.65 |128.4 |4.9 |
| SAC |0.17 |-64.0 |128.5 |

## Resources

- [Keke Competition](http://keke-ai-competition.com/)
- [baba-is-auto](https://github.com/utilForever/baba-is-auto)
- [Soft Actor Critic Tutorial](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC/tf2)
