import os
import yaml
import numpy as np
from ast import literal_eval
from pymongo import MongoClient

class BaseConnection:
    '''Get a connection to a mongo database server'''
    
    def establish_db_connection(self, prefix):
        '''
        Parameters:
        -----------
        prefix : str
            Which database service to connect to, references yaml config
        '''
        client = MongoClient(host=self.experiment['{}_host'.format(prefix)],
                     ssl=True,
                     ssl_ca_certs=self.experiment['{}_ssl_ca_file'.format(prefix)],
                     port=self.experiment['{}_port'.format(prefix)])
        
        
        self.db = client[self.experiment['{}_database'.format(prefix)]]
        self.col = self.db[self.experiment['{}_collection'.format(prefix)]]        


class BaseReader:
    '''Base class for reading and manipulating an experiment yaml config file.'''
    
    def get_config(self, config):
        '''Read yaml config file, and create the experiment dictionary, and hyperparameter bounds.'''

        self.config_dict = yaml.load(open(config, 'r'))
        self._bundle_experiment()
        self._bundle_hyperparameters()


    def _bundle_experiment(self):
        ''' Merge a dictionary of dictionaries into a single dictionary. Don't include the hyperparameters. '''
        
        config_keys = list(self.config_dict.keys())
        
        # Remove hyperparameters from the experiment configutation
        if 'hyperparameters' in config_keys:
            config_keys.pop(config_keys.index('hyperparameters'))
            
        experiment = self.config_dict[config_keys[0]]
        
        # Flatten the experiment headings
        for d in config_keys[1:]:
            experiment = self._merge_two_dicts(experiment, self.config_dict[d])
        self.experiment = experiment

        self._fix_none()


    def _merge_two_dicts(self, x, y):
        ''' Merge two dictionaries into one.'''
        z = x.copy()
        z.update(y)
        return z


    def _fix_none(self):
        ''' When the yaml file is read, None values are read as strings. Change the type to python None.'''
        
        for k,v in self.experiment.items():
            if v == 'None':
                self.experiment[k] = None


    def _bundle_hyperparameters(self):
        ''' This method accomplishes two import tasks.
             
             (1) It reformats the the hyperparameters into a list of dictionaries 
             called 'bounds', which is in the format required for input to GPyOpt. 
             
             (2) It creates the list 'hyperparameter_names' which tracks the name 
             of each dictionary in 'bounds'. This is needed because GpyOpt only passes 
             as it's recommendation, a single list of variables. In order to properly 
             place them into the 'experiment' dictionary, we need to to know which 
             positional element in the recommendation is which. I.e. 'bayesian_names' 
             tells us the mapping of the recommendation from GPyOpt to the named the named
             element in 'experiment'.
        '''
        
        # Get the parameters for hyperparameter optimization
        hyperparameters = self.config_dict['hyperparameters']
        
        # Initialize a list of dictionaries to pass to GPyOpt
        self.bounds = []
        
        # Keep a list of element names for each element in bounds to map back into 'experiment' later
        self.hyperparameter_names = []
        
        # Accumulate the hyperparameter keys/values
        for k,v in hyperparameters.items():
            if 'param' in k:
                # the tuple is read as a string during the yaml read, change it back to a tuple
                v['domain'] = literal_eval(v['domain'])
                
                self.hyperparameter_names.append(v['name'])
                self.bounds.append(v)
                
        # Re-Order the hyperparameters in case a specific order is required to construct a model
        if 'read_order' in self.experiment.keys():
            ordered_names = []
            ordered_bounds = []
            for r in self.experiment['read_order']:
                ordered_names.append(r)

                idx = self.hyperparameter_names.index(r)
                ordered_bounds.append(self.bounds[idx])

            self.hyperparameter_names = ordered_names
            self.bounds = ordered_bounds
            
            
    