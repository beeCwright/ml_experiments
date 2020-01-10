import os
import yaml
import inspect
import numpy as np
import tensorflow as tf
from .base import BaseReader, BaseConnection


class ModelModifier:
    def __init__(self, get_model_func, experiment=None):
        '''
        Description:
        ------------
        Base class methods for modifying and manipulating a keras model
        
        
        Params:
        -------
        get_model_func (func): function the returns a keras model
        experiment (dict): key/values needed when calling get_model_func
        '''
        
        self.get_model_func = get_model_func
        self.experiment = experiment
        
        
    def get_network_signature(self):
        '''
        Description:
        ------------
        Get the network call signature keys and values
        '''
        
        self.args = inspect.signature(self.get_model_func)
        self.model_signature = list(self.args.parameters.keys())


    def fill_default_parameters(self):
        '''
        Description:
        ------------
        Fill in the parameters with the default values found in the model signature
        '''
        self.param_dict = dict()
        
        params = self.args.parameters.values()
        for p in params:
            if p.default != inspect._empty:
                self.param_dict[p.name] = p.default
            else:
                self.param_dict[p.name] = None
        
    
    def find_matching_parameters(self):
        '''
        Description:
        ------------
        Match any parameters in the config that are in the signature
        '''
        for param in self.model_signature:
            if param in self.experiment.keys():
                self.param_dict[param] = self.experiment[param]    
    
    def get_model(self):
        '''
        Description:
        ------------
        Call the networks initalization method using matching parameters from the experiment
        
        Return:
        -------
        model (keras model)
        '''
    
        model = self.get_model_func(**self.param_dict)
        return model
    
    
class ModelHandler(BaseReader, BaseConnection):
    '''Load model checkpoints, set cuda environment variables.'''
    
    def __init__(self, config_path, experiment_name, prefix='manager'):
        self.get_config(config_path) # inherited method, read yaml config
        self.establish_db_connection(prefix=prefix) # inherited method, connect to mongo
        self.experiment_name = experiment_name


    def get_experiment_entry(self):
        '''Get the experiment entry from mongo.'''
        self.entry = db.col.find_one({'experiment_name' : self.experiment_name})

        
    def set_gpus(self, gpu_num=3):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        
    
    def retrieve_model(self, step=None, gpu_num=3):
        '''
        Description: Load the model and parallize the gpu's
        '''
        self.get_checkpoint(self.entry['model_dir'], step=step)
        self.set_gpus(gpu_num)
        self.model = tf.keras.models.load_model(self.checkpoint)


    def get_checkpoint(self, path, step=None):
        ''' Get the best model checkpoint based on validation loss.

        Parameters:
        -------
        path : str
            absolute file path to the model checkpoint directory
        step : str
            specific model step to load
        '''

        if step == None:
            checkpoints = os.listdir(path)
            loss = [float(c.split('--')[1].split('.hdf5')[0]) for c in checkpoints]
            loss = np.array(loss)
            idx = np.argsort(loss)
            best_checkpoint = checkpoints[idx[-1]]
            self.checkpoint = os.path.join(path, best_checkpoint)
        else:
            self.checkpoint = os.path.join(path, step)
