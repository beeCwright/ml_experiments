import GPy, GPyOpt
import numpy
import numpy as np
from .base import BaseReader, BaseConnection


class ExperimentController(BaseReader, BaseConnection):
    ''' Class that instantiates a controller client to make and receive experiment updates.'''

    def __init__(self, config_path, ignored_experiments=None):
        ''' Create a controller to maintain a gpu worker node that enters a pool of worker nodes.
        
        Parameters
        ----------
        experiment : dict 
            All experiment parameters as a dictionary
        bounds : list
            A list of arrays (2D). These are the hyperparameters for GPyOpt. Rows are the combinations of encoded hyperparameters.
            
        '''
        self.get_config(config_path) # inherited method, read yaml config

        experiment_keys = list(self.experiment.keys())
        assert 'max_iter' in experiment_keys, 'max_iter is a required parameter in the .yaml config.'
        assert 'max_local_iter' in experiment_keys, 'max_local_iter is a required parameter in the .yaml config.'
        assert 'num_warm_up' in experiment_keys, 'num_warm_up is a required parameter in the .yaml config.'

        self.max_iter = self.experiment['max_iter']
        self.max_local_iter = self.experiment['max_local_iter']
        self.num_warm_up = self.experiment['num_warm_up']
        
        self.establish_db_connection(prefix='controller') # inherited method, connect to mongo
        self._get_design()
        self.ignored_experiments = ignored_experiments
        
        
    def _get_design(self):
        ''' Get the latest updated experiment. '''
        
        # Retrieve the existing design
        if self.col.estimated_document_count() == 1:
            self.raster = self.col.find_one()
            self.raster_id = self.raster['_id']
            self.X_steps = self.raster['X_steps']
            self.Y_steps = self.raster['Y_steps']
            self.next_iter = self.raster['next_iter']
            self.warm_up_list = self.raster['warm_up_list']
            
        # Create a new design
        elif self.col.estimated_document_count() == 0:
            self._create_design_entry()

        else:
            raise Exception('There should only be one document in the controller! Review database {}'.format(self.experiment['controller_database']))
            
            
    def _create_design_entry(self):
        ''' Create the controller document in mongo. '''
        
        # Determine the dimensionality of each hyperparameter
        experiment_dimensions = []        
        for i in self.bounds:
            experiment_dimensions.append(len(i['domain'])) # list of counts of dimensions per hyperparameter
            
        warm_up_array = np.zeros((self.num_warm_up, len(experiment_dimensions)))
        
        # Populate the warm up table (each row is an experiment, each column is a hyperparameter)
        for i in range(self.num_warm_up):
            for j, dim in enumerate(experiment_dimensions):

                # Draw a random sample for this hyperparameter from its domain
                this_dimensions_sample = np.random.choice(dim, 1)
                warm_up_array[i][j] = self.bounds[j]['domain'][this_dimensions_sample[0]]
                
        # Make it into a list
        self.X_steps = []
        self.Y_steps = []
        self.warm_up_list = []
        for i in warm_up_array:
            self.warm_up_list.append(list(i))
            
        # Insert into mongo
        self.next_iter = 0
        self.entry_id = self.col.insert_one({'next_iter' : 0,
                                'num_warm_up' : self.num_warm_up,
                                'warm_up_list' : self.warm_up_list,
                                'X_steps' : self.X_steps,
                                'Y_steps' : self.Y_steps})
        
        # Keep the ID for reference
        self.raster_id = self.entry_id.inserted_id


    def _tally_an_iteration(self):
        ''' Update the experiments completed tally locally and globally.'''

        self.next_iter += 1
        self._set_val('next_iter', self.next_iter)
    
    
    def _checkout_warmup(self):
        ''' Get an experiment from the warm up raster.

        Returns
        -------
        warm_up_entry : array
            1D array of encoded hyperparameter combinations
        '''
        
        warm_up_entry = self.warm_up_list[self.next_iter]
        return warm_up_entry
        
        
    def _do_bayesian_optimization(self):
        ''' Use GPyOpt to perform bayesian optimization on a set of parameters
        
        Returns
        -------
        next_trial : array
            1D array of encoded hyperparameter combinations
        '''
        assert self.X_steps != [], 'X_steps cannot be an empty list'
        assert self.Y_steps != [], 'Y_steps cannot be an empty list'

        b_opt = GPyOpt.methods.BayesianOptimization(f=None, 
                                            domain=self.bounds, 
                                            X = np.array(self.X_steps),
                                            Y = self.Y_steps)
        
        x_next = b_opt.suggest_next_locations(ignored_X=self.ignored_experiments)
        next_trial = x_next[0]
        return next_trial
    

    def _set_val(self, key, value):
        ''' Set a value on the controller mongo document
        
        Parameters
        ----------
        key : str
            Element in the controller mongo document to be updated
        value : array/int/float
            Any valid mongo type, the updated value the element being set
        '''
        self.col.update_one({'_id' : self.raster_id}, {'$set' : {key : value}})


    def update_design(self, x_step, y_step):
        ''' Update the controller database with the latest values.
        
        Parameters
        ----------
        x_step : array
            1D array of encoded hyperparameter combinations
        y_step: list
            value (loss) of objective function being optimzied over
        '''
        assert type(y_step) == list
        assert type(x_step) == numpy.ndarray

        self._get_design()

        self.X_steps.append(list(x_step))
        self._set_val('X_steps', self.X_steps)

        self.Y_steps.append(y_step)
        self._set_val('Y_steps', self.Y_steps)


    def get_next_suggestion(self):
        ''' Get the next hyperparameters either from the warmup or by requesting it from GPyOpt.
        
        Returns
        -------
        next_trial : array
            1D array giving the encoded hyperparamters for the next experiment
        '''
        
        # Get the latest design of executed experiments
        self._get_design()
        
        # If we're still warming up, get an experiment from the warmup
        if self.next_iter+1 <= self.num_warm_up:
            warm_up_str = 'Getting warmup trial: (' + str(self.next_iter+1) + '/' + str(self.num_warm_up) + ')'
            print(warm_up_str)
            self.next_trial = self._checkout_warmup()
            
        # Otherwise perform bayesian optimization to get the next recommended experiment
        else:
            trial_str = 'Getting trial: (' + str(self.next_iter+1 ) + '/' + str(self.max_iter) + ')'
            print(trial_str)
            self.next_trial = self._do_bayesian_optimization()

        # Update the number of experiments that have been tried to the database server as well as locally
        self._tally_an_iteration()
        return self.next_trial