# ml-experiments
This repository contains python code for managing experiment recommendations. It is built on top of GPyOpt, to perform Bayesian Optimizastion over a set of hyperparameters. It uses MongoDB as a central store of all experiments and results, allowing any number to nodes to asynchronously join an experiment and to checkout and evaluate trials.

---


### Installation

Install the GPyOpt repository, this requires you have gcc installed.

```
$ pip3 install git+https://github.com/SheffieldML/GPyOpt.git
```


### Usage

 - Create a controller instance which will read a yaml configuration file, and get the latest experiment updates, or create an experiment entry is none exists.

``` python
import ml_experiments as me

config = './ml_experiments/demo_config_small.yaml'
controller = me.ExperimentController(config)
```

 - The controller bundles the hyperparameters into the format GPyOpt requires.

``` python
controller.bounds
[{'name': 'learning_rate', 'type': 'discrete', 'domain': (0.005, 0.001)},
 {'name': 'num_convs', 'type': 'discrete', 'domain': (1, 2)},
 {'name': 'network', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
 {'name': 'dropout', 'type': 'discrete', 'domain': (0, 0.1, 0.2)},
 {'name': 'kernel', 'type': 'discrete', 'domain': (3, 5)},
 {'name': 'normalize', 'type': 'discrete', 'domain': (0, 1)},
 {'name': 'vertical_flip', 'type': 'discrete', 'domain': (0, 1)}]
```

 - Upon experiment creation, a set of warm-up trials are generated and used as the initial recommendations.  
 - After `num_warm_up` iterations have passed, the GPyOpt library is used to recommend the next trial.

``` python
current_iter = controller.next_iter
while (current_iter < controller.max_iter) and (local_iter < controller.max_local_iter):

    # Get the next trial, and increment the global next_iter
    this_trial = controller.get_next_suggestion()

    # Evaluate (custom code goes here)
    loss = evaluate(this_trial)

    # Send the update back to the controller
    controller.update_design(this_trial, [loss])

    # Claim the next iteration for this local instance
    current_iter = controller.next_iter
    local_iter += 1
```