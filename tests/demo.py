import numpy as np
import tensorflow as tf
import ml_experiments as me
from heka.models.networks.unet import UNetConfig, get_unet

# To run move this file so the ml_experiments is a child directory

def get_model(experiment, get_model_func):
    '''Get a UNet using the experiment parameters.'''
    
    # Get the model, match parameters from experiment with the func signature, otherwise use the defaults
    with tf.device('/cpu:0'):            
        mm = me.ModelModifier(get_model_func, experiment)
        mm.get_network_signature()
        mm.fill_default_parameters()
        mm.find_matching_parameters()
        model = mm.get_model()
        model = get_unet(model)
    return model


def set_trial(x, experiment, hyperparameter_names):
    ''' GPyOpt recommends experiments to be performed next to guide 
    hyperparameter searches. The next recommened experiment is 'x'. 
    This method takes 'x', and sets the values in the 'experiment'. 
    'x' is a list, 'experiment' is a dictionary.

    Params:
    -------
    x (list) : Values recommended by Bayesian Optimizer for next experiment.
    '''
    for c, val in enumerate(x):
        # Network depth and feature maps
        if hyperparameter_names[c] == 'network':
            val = int(val)
            compression_arms = [
                [32, 64, 128, 256],
                [32, 64, 128, 256, 512],                
                [32, 48, 64, 80],
                [32, 48, 64, 80, 96],                
                [32, 48, 64, 80, 96, 112],
            ]

            decompression_arms = [
                [128, 64, 32],                
                [256, 128, 64, 32],
                [64, 48, 32],
                [80, 64, 48, 32],
                [96, 80, 64, 48, 32],
            ]

            experiment['compression_channels'] = compression_arms[val]
            experiment['decompression_channels'] = decompression_arms[val]

        # Dropout
        elif hyperparameter_names[c] == 'dropout':
            experiment['compression_dropout'] = [val] * len(experiment['compression_channels'])
            experiment['decompression_dropout'] = [val] * len(experiment['decompression_channels'])

        # Scheduler
        elif hyperparameter_names[c] == 'decay_rate':
            experiment['decay_rate'] = val

        elif hyperparameter_names[c] == 'kernel':
            if experiment['dimension'] == '3D':
                experiment['kernel_size'] = [int(val), int(val), int(val)]
            elif experiment['dimension'] == '2D':
                experiment['kernel_size'] = [int(val), int(val)]

       # Learning Rate
        elif hyperparameter_names[c] == 'learning_rate':
            experiment['learning_rate'] = val
                
        # Everything else
        else:
            val = int(val)
            experiment[hyperparameter_names[c]] = val
        
        print('{:4} {} {}'.format('', hyperparameter_names[c], val))
    return experiment


def get_trial(one_trial):
    ''' Conduct a full training run of a model given the above parameters.'''
    
    # Create an experiment based on a recommended trial
    this_experiment = set_trial(one_trial, controller.experiment, controller.hyperparameter_names)
    
    # Match the parameters in experiment with parameters in the model func signature
    model = get_model(this_experiment, UNetConfig)
    
    
    # Run your model
    # val_loss = eval_trial(model)
    val_loss = [4,5,3,10,4,3,2,1,1,1]
    
    # Lightly smooth the loss to avoid large spikes
    f = 3
    val_loss = np.convolve(np.array(val_loss), np.ones(f)/f)[:-(f-1)]

    # Record the minimum and return it to GPyOpt
    loss = np.min(val_loss)
    
    return loss



if __name__ == '__main__':

    config = './ml_experiments/demo_config.yaml'
    config = './ml_experiments/demo_config_small.yaml'
    controller = me.ExperimentController(config)

    local_iter = 0
    current_iter = controller.next_iter
    while (current_iter < controller.max_iter) and (local_iter < controller.max_local_iter):

        # Get the next trial, and increment the global next_iter
        this_trial = controller.get_next_suggestion()

        # Evaluate
        loss = get_trial(this_trial)

        # Send the update back to the controller
        controller.update_design(this_trial, [loss])

        # Claim the next iteration for this local instance
        current_iter = controller.next_iter
        local_iter += 1
