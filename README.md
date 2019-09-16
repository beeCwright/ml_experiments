# ml-experiments
A repository for managing experiment recommendataions built on top of GPyOpt.

___

### Installation

Install the GPyOpt repository, this requires you have gcc installed.

```pip3 install git+https://github.com/SheffieldML/GPyOpt.git```


### Usage

Create a controller instance which will read a yaml configuration file, and get the latest experiment updates, or create an experiment entry is none exists.

```
import ml_experiments as me

config = './ml_experiments/demo_config_small.yaml'
controller = me.ExperimentController(config)

```

The controller bundles the hyperparameters into the format GPyOpt requires.
```
controller.bounds
[{'name': 'learning_rate', 'type': 'discrete', 'domain': (0.005, 0.001)},
 {'name': 'num_convs', 'type': 'discrete', 'domain': (1, 2)},
 {'name': 'network', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
 {'name': 'dropout', 'type': 'discrete', 'domain': (0, 0.1, 0.2)},
 {'name': 'kernel', 'type': 'discrete', 'domain': (3, 5)},
 {'name': 'normalize', 'type': 'discrete', 'domain': (0, 1)},
 {'name': 'vertical_flip', 'type': 'discrete', 'domain': (0, 1)}]
```

Upon experiment creation, an initial set of warm-up trials are generated and used as recommendations. After `max_iter` iterations have passed, the GPyOpt library is used to recommend the next trial.

```
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