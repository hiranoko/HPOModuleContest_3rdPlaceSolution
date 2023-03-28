# HPO Module Contest Solution Description

This repository contains 3rd-solution for the [HPO Module Contest](https://signate.jp/competitions/978).

This contest presents the following challenges:

1. Limited trials (100)
2. A large number of parameters (10-20)
3. Vast search space.

My solution is based on the TuRBO algorithm, which balances exploration and exploitation by narrowing down the search space based on the number of updates of the best parameters. I also added a restart feature to monitor the number of updates and implemented a forced restart when the same evaluation value continues. These enhancements were added to [aiaccel](https://github.com/aistairc/aiaccel) for a more effective hyperparameter optimization process.

## Key Features

- Restart functionality using the TuRBO algorithm
- Flexibility to change kernel functions and probability models in botorch
- Implemented as a sampler in Optuna, enabling tell-and-ask functionality

## Directory structure

```terminal
HPOModuleContest_3rdPlaceSolution
└── src
    └── workspace
        ├── model # optimizer parameters
        ├── src   # optimizer main unit
        └── tests # benchmark function
            └── schwefel_5dim
```

## Installation

```terminal
$ pip install git+https://github.com/aistairc/aiaccel.git
$ pip install -r requirement.txt
```

## Usage

```terminal
$ cd src
$ bash local_run.sh
```

## Acknowledgement

The codes are based on [BoTorch](https://github.com/pytorch/botorch) and [optuna](https://github.com/optuna/optuna). Please also follow their licenses. Thanks for their awesome works.
