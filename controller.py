import GPy, GPyOpt
import numpy as np
from .base import BaseReader, BaseConnection


class ExperimentController(BaseReader, BaseConnection):
    '''
    Description: Methods for coordinating multiple worker nodes training from the same hyperparameter set.
    Created: 19/07/12
    Modified: 19/08/05
    '''
    
    def __init__(self, config_path, ignored_experiments=None):
        '''
        Description: Establish a controller to maintain workers who enter the pool.
        ------------
        
        Params:base
        -------
        experiment (dict): all experiment parameters as a dictionary
        bounds (list): a list of arrays (2D). These are the hyperparameters for GPyOpt. Every row are the encoded hyperparameters.
        '''
        self.get_config(config_path)

        self.max_iter = self.experiment['max_iter']
        self.num_warm_up = self.experiment['num_warm_up']
        self.establish_db_connection()
        self.get_design()
        self.ignored_experiments = ignored_experiments
        
        
    def get_design(self):
        '''
        Description: Get a list of all input and output experiment encodings    
        '''
        
        # Retrieve existing design
        if self.db.col.count() > 0:
            self.raster = self.db.col.find_one()
            self.warm_up_list = self.raster['warm_up_list']
            self.raster_id = self.raster['_id']
            self.next_iter = self.raster['next_iter']
            self.X_steps = self.raster['X_steps']
            self.Y_steps = self.raster['Y_steps']

        # Create a new design
        else:
            self.create_design_entry()

            
    def create_design_entry(self):
        '''
        Description: Create the controller document in mongo
        '''
        
        # Create a blank slate of warm up experiments
        experiment_dimensions = []        
        for i in self.bounds:
            experiment_dimensions.append(len(i['domain']))
        
        # Create twice as many as are needed incase a node fails and we need backups
        twice_as_many_needed_wamrups = self.num_warm_up*2
        warm_up_array = np.zeros((twice_as_many_needed_wamrups, len(experiment_dimensions))) # create twice as many as needed
        
        # Populate the warm up table with random configurations
        for i in range(twice_as_many_needed_wamrups):
            for j,dim in enumerate(experiment_dimensions):
                this_dimensions_sample = np.random.choice(dim, 1)
                warm_up_array[i][j] = self.bounds[j]['domain'][this_dimensions_sample[0]]
                
        # Make it into a list
        self.X_steps = []
        self.Y_steps = []
        self.warm_up_list = []
        for i in warm_up_array:
            self.warm_up_list.append(list(i))
            
        # Insert the design into the mongo database
        self.next_iter = 0
        self.entry_id = self.db.col.insert_one({'next_iter' : 0,
                                'num_warm_up' : self.num_warm_up,
                                'warm_up_list' : self.warm_up_list,
                                'X_steps' : self.X_steps,
                                'Y_steps' : self.Y_steps})
        
        # Keep the ID for reference
        self.raster_id = self.entry_id.inserted_id
        

    def get_next_suggestion(self):
        '''
        Description: Get the next experiment either from the warmup or by requesting it form GPyOpt
        
        Returns:
        --------
        next_trail (array): the output from GPyOpt, and 1D array giving the encoded hyperparamters for the next experiment
        '''
        
        # Get the latest design of executed experiments
        self.get_design()
        
        # If we're still warming up, get an experiment from the warmup
        if self.next_iter+1 <= self.num_warm_up:
            warm_up_str = 'Getting warmup trial: (' + str(self.next_iter+1) + '/' + str(self.num_warm_up) + ')'
            print('{:12} {}'.format('', warm_up_str))
            self.next_trial = self.checkout_warmup()
            
        # Otherwise perform bayesian optimization to get the next recommended experiment
        else:
            trial_str = 'Getting trial: (' + str(self.next_iter+1 ) + '/' + str(self.max_iter) + ')'
            print('{:12} {}'.format('', trial_str))
            self.next_trial = self.do_bayesian_optimization()            
            
        # Update the number of experiments that have been tried to the database server as well as locally
        self.tally_an_iteration()
        return self.next_trial
    

    def tally_an_iteration(self):
        '''
        Description: Keep a local tally, as well as updating the global tally on the database server for the number of experiments that have been executed
        '''
        self.next_iter += 1
        self.set_val('next_iter', self.next_iter)
    
    
    def checkout_warmup(self):
        '''
        Description: Get an experiment from the warm up raster
        '''
        # Return the next warmup trial
        return self.warm_up_list[self.next_iter]
        
        
    def do_bayesian_optimization(self):
        '''
        Description: Use GPyOpt to perform bayesian optimization on a set of parameters, and return the next experiment
        '''
        b_opt = GPyOpt.methods.BayesianOptimization(f=None, 
                                            domain=self.bounds, 
                                            X = np.array(self.X_steps),
                                            Y = np.array(self.Y_steps))
        
        x_next = b_opt.suggest_next_locations(ignored_X=self.ignored_experiments)
        
        return x_next[0]
    

    def update_design(self, x_step, y_step, current_iter):
        '''
        Description: Update the controller database with the latest values (input and output) from an experiment
        '''
        self.get_design()

        self.X_steps.append(list(x_step))
        self.set_val('X_steps', self.X_steps)

        self.Y_steps.append(y_step)
        self.set_val('Y_steps', self.Y_steps)

    
    def set_val(self, key, value):
        '''
        Description: Set a value on the controller node
        '''
        self.db.col.update_one({'_id' : self.raster_id}, {'$set' : {key : value}})