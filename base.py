import os
import yaml
import numpy as np
from ast import literal_eval
from pymongo import MongoClient

class BaseConnection:
    def establish_db_connection(self):
        '''
        Description: Get a connection to the controller mongo database server
        '''
        client = MongoClient(host=self.experiment['controller_host'],
                     ssl=True,
                     ssl_ca_certs=self.experiment['controller_ssl_ca_file'],
                     port=self.experiment['controller_port'])
        
        
        self.db = client[self.experiment['controller_database']]
        self.col = self.db[self.experiment['controller_collection']]        


class BaseReader:
    '''
    Description: Base class for the network and data boilers.
    Created: 18/11/28
    Modified: 19/6/05
    '''
        
    def bundle_experiment(self):
        '''
        Description: Merge a dictionary of dictionaries into a single dictionary. Don't include the bayesian parameters.        
        '''
        config_keys = list(self.config_dict.keys())
        
        # Don't include bayesian parameters in the experiment, they have a special format
        if 'bayesians' in config_keys:
            config_keys.pop(config_keys.index('bayesians'))
            
        experiment = self.config_dict[config_keys[0]]
        def _merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z
        for d in config_keys[1:]:
            experiment = _merge_two_dicts(experiment, self.config_dict[d])
        self.experiment = experiment
        
        self.experiment = self.fix_none(experiment)


    def fix_none(self, experiment):
        '''
        Description: When the yaml file is read in None values are read as strings. Change the type to python None
        '''
        for k,v in experiment.items():
            if v == 'None':
                experiment[k] = None
        return experiment


    def get_config(self, config):
        '''
        Description: Read configuration parameters from yaml.
        '''
        self.config_dict = yaml.load(open(config, 'r'))
        self.bundle_experiment()
        self.unbundle_bayesians()


    def unbundle_bayesians(self):
        '''
        Description: This method accomplishes two import tasks. (1) It reformats the the 
                     bayesian parameters into a list of dictionaries called 'bounds', 
                     which is in the format needed by GPyOpt. (2) It creates the list
                     'bayesian_names' which tracks the name of each dictionary in 'bounds'.
                     This is needed because GpyOpt only passes as it's recommendation, a 
                     single list of variables. In order to properly place them into the 
                     'experiment' dictionary, we need to to know which positional element
                     in the recommendation is which. I.e. 'bayesian_names' tells us the
                     mapping of the recommendation from GPyOpt to the named the named
                     element in 'experiment'.
        '''
        # Get the parameters for bayesian optimization
        bayesians = self.config_dict['bayesians']
        
        # Initialize a list of dictionaries to pass to GPyOpt
        self.bounds = []
        
        # Keep a list of element names for each element in bounds to map back into 'experiment' later
        self.bayesian_names = []
        
        for k,v in bayesians.items():
            if 'param' in k:
                # the tuple is read as a string during the yaml read, change it back to a tuple
                v['domain'] = literal_eval(v['domain'])
                
                self.bayesian_names.append(v['name'])
                self.bounds.append(v)
                
        # Often certain network parameters need to be set in a specific order
        # ex) dropout must be set after the compression/decompression channels
        # re-order the variables for GpyOpt in a user specified order
        if 'read_order' in self.experiment.keys():
            ordered_names = []
            ordered_bounds = []
            for r in self.experiment['read_order']:
                ordered_names.append(r)

                idx = self.bayesian_names.index(r)
                ordered_bounds.append(self.bounds[idx])

            self.bayesian_names = ordered_names
            self.bounds = ordered_bounds
            
            
    