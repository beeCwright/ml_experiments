import datetime
import numpy as np
import pkg_resources
import pandas as pd
from .base import BaseConnection, BaseReader
from tensorflow.keras.callbacks import Callback


class ExperimentNamer:
    '''Class methods for naming experiments.'''
    
    def __init__(self):
        '''Get a list of experiment names.'''
        path = '/names.csv'
        filepath = pkg_resources.resource_filename(__name__, path)
        self.name_pool = list(pd.read_csv(filepath)['name'].values)
    
    
    def get_used_names(self, df):
        '''Get a list of all names in the field 'name' in a pandas dataframe.

        Parameters:
        -----------
        df (pd.DataFrame): dataframe where each row is an experiment

        '''

        # There are no previous experiment names
        if len(df) == 0:
            self.used_names = []
        else:
            self.used_names = list(df.experiment_name.values)

    
    def get_unused_names(self):
        '''
        Get a list a names from the name pool that aren't in the used names.
        '''
        
        self.unused_names = [name for name in self.name_pool if name not in self.used_names]
        
        # If all names are used, add a suffix to create an additional 1000 names
        if len(self.unused_names) == 0:
            for i in range(10):
                suffix = '_' + str(i)
                self.unused_names = [name + suffix for name in self.name_pool]


    def get_random_unused_name(self):
        ''' Draw a random name that isn't already used, and pop it from the list.'''
        
        rand_idx = np.random.choice(len(self.unused_names), 1)[0]
        random_name = self.unused_names.pop(rand_idx)
        return random_name
    
    
class ExperimentRecorder(ExperimentNamer, BaseReader, BaseConnection, Callback):
    '''Class methods for recording experiments in mongo.'''
        
    def __init__(self, config_path):
        
        # Instantiate the experiment namer and name pool
        ExperimentNamer.__init__(self)

        # Get the experiment configuration inherited from BaseReader
        self.get_config(config_path)

        # Establish connection with mongoDB
        self.establish_db_connection('manager')
        
        # Establish which epoch to start recording values
        assert 'start_recording' in list(self.experiment.keys()), 'A start_recording value must be included in the experiment yaml.'
        self.start_recording = self.experiment['start_recording']

        self.results = {}
        for name in self.experiment['train_metrics']:
            self.results[name] = []
            self.results['val_' + name] = []


        # Get any previous experiment names
        previous_experiments = list(self.col.find())
        df = pd.DataFrame(previous_experiments)

        # Determine which names have been used
        self.get_used_names(df)
        self.get_unused_names()

        # Draw a random name to call this experiment
        this_experiment_name = self.get_random_unused_name()
        self.experiment['experiment_name'] = this_experiment_name
        print('{:12} {} {}'.format('', 'this experiment is called: ', this_experiment_name))
        print('{:12} {} {} {}'.format('', 'WARNING', len(self.unused_names), 'experiment names remaining'))

        # Initalize empty lists for loss values
        self.loss = []
        self.val_loss = []


    def _merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def create_experiment_entry(self):
        date = {'date_created' : datetime.datetime.utcnow()}
        experiment = self._merge_two_dicts(self.experiment, date)
        self.entry_id = self.col.insert_one(experiment)
        print('Created experiment entry: ', self.entry_id.inserted_id)

    def update_experiment_results(self, results):
        self.col.update_one({'_id' : self.entry_id.inserted_id}, {'$set' : results})
        print('Experiment results succesfully recorded.')

        
    def on_epoch_begin(self, epoch):
        if epoch == self.start_recording:
            self.create_experiment_entry()

            
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= self.start_recording:
            for name in self.experiment['train_metrics']:
                self.results[name].append(np.float64(logs[name]))
                self.results['val_'+name].append(np.float64(logs['val_'+name])) 

            self.update_experiment_results(self.results)