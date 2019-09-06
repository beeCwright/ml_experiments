from .base import BaseConnection, BaseReader
from tensorflow.keras.callbacks import Callback


class ExperimentNamer:
    '''
    Desription: Class methods for naming experiments.  
    Created: 18/11/21
    Modified: 19/07/17
    '''
    
    def __init__(self):
        '''
        Description: 
        '''
        # Instantiate the pool of names
        self.name_pool = ['ampere', 'aristotle', 'avogadro', 'bell', 'bengio', 'bernoulli', 'bohr', 
                         'boyle', 'braun', 'bunson', 'curie', 'dalton', 'darwin', 'dirac', 'edison', 
                         'einstein', 'euclid', 'euler', 'faraday', 'fermi', 'feynman', 'fisher', 
                         'fleming', 'gauss', 'gibbs', 'hawking', 'heisenberg', 'hertz', 'hewish', 
                         'hilbert', 'hinton', 'hubble', 'hodgkin', 'hooke', 'hypatia', 'ising', 'joule', 
                         'kaku', 'karpathy', 'kepler', 'kirchoff', 'lagrange', 'lamarck', 'laplace', 
                         'lecun', 'leibniz', 'maxwell', 'mendel', 'minsky', 'nakaya', 'natta', 'newton',
                         'nightingale', 'nobel', 'nye', 'ohm', 'oppenheimer', 'pascal', 'planck', 
                         'pythagoras', 'ray', 'riemann', 'sagan', 'schottky', 'schrodinger', 'seaborg',
                         'somerville', 'steno', 'shoemaker', 'simpson', 'stevens', 'tartaglia', 'teller',
                         'tesla', 'thompson', 'toricelli', 'townes', 'turing', 'tyson', 'urey', 'venter', 
                         'virchow', 'volta', 'wald', 'wallace', 'watt', 'wheeler', 'willis', 'wilson', 'wright']
    
    
    def get_used_names(self, df):
        '''
        Description: Get a list of all names in the field 'name' in a pandas dataframe
        '''
        if len(df) == 0:
            # There are no previous experiment names
            self.used_names = []
        else:
            self.used_names = list(df.experiment_name.values)

    
    def get_unused_names(self):
        '''
        Description: Get a list a names from the name pool that aren't in the used names
        '''
        
        self.unused_names = [name for name in self.name_pool if name not in self.used_names]
        
        # If all names are used, add a suffix to create an additional 1000 names
        if len(self.unused_names) == 0:
            for i in range(10):
                suffix = '_' + str(i)
                self.unused_names = [name + suffix for name in self.name_pool if name + suffix not in self.used_names]
    
        
        
    def get_random_unused_name(self):
        '''
        Description: Draw a random name that isn't already used, and pop it from the list
        '''
        rand_idx = np.random.choice(len(self.unused_names), 1)[0]
        random_name = self.unused_names.pop(rand_idx)
        return random_name
    
    
class ExperimentRecorder(Callback, ExperimentNamer, BaseReader, BaseConnection):
    '''    
    Desription: Class methods for recording experiments in mongo.
    Created: 18/10/18
    Modified: 19/08/05
    '''
        
    def __init__(self, config_path):
        ExperimentNamer.__init__(self)

        self.get_config(config_path)

        # Establish connection with mongoDB
        self.establish_db_connection()        
        
        self.start_recording = sefl.experiment['start_recording']
        self.train_metric_name = None
        
        # Setup the ExperimentNamer to name this experiment        
        if self.experiment['namer']:

            # Get any previous experiment names
            previous_experiments = list(self.db.col.find())   # all training runs in the trial
            df = pd.DataFrame(previous_experiments)   # store into a dataframe

            # Determine which names have been used
            self.get_used_names(df)
            self.get_unused_names()

            # Draw a random name to call this experiment
            this_experiment_name = self.get_random_unused_name()
            self.experiment['experiment_name'] = this_experiment_name            
            print('{:12} {} {}'.format('', 'this is experiment is called: ', this_experiment_name))
            print('{:12} {} {} {}'.format('', 'WARNING', len(self.unused_names), 'experiment names remaining'))

        # Initalize empty lists for loss values
        self.loss = []
        self.val_loss = []

        if experiment['train_metric_name'] != None:
            self.train_metric_name = experiment['train_metric_name']
            self.val_metric_name = 'val_' + experiment['train_metric_name']
            self.train_metric = []
            self.val_metric = []

    def _merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def create_experiment_entry(self):
        date = {'date_created' : datetime.datetime.utcnow()}
        experiment = self._merge_two_dicts(self.experiment, date)
        self.entry_id = self.db.col.insert_one(experiment)
        print('Created experiment entry: ', self.entry_id.inserted_id)

    def update_experiment_results(self, results):
        self.db.col.update_one({'_id' : self.entry_id.inserted_id}, {'$set' : results})
        print('Experiment results succesfully recorded.')

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == self.start_recording:
            self.create_experiment_entry()

    def on_epoch_end(self, epoch, logs={}):
        
        # Minimally record the loss
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        # Optionally record metrics
        if self.experiment['train_metric_name'] != None:
            self.train_metric.append(logs.get(self.train_metric_name))  # categorical_accuracy
            self.val_metric.append(logs.get(self.val_metric_name))      # 'val_categorical_accuracy'

        if epoch >= self.start_recording:
            if self.train_metric_name != None:
                results = {'loss' : self.loss, 
                           'val_loss' : self.val_loss, 
                           self.train_metric_name : self.train_metric, 
                           self.val_metric_name : self.val_metric}
            else:
                results = {'loss' : self.loss, 
                           'val_loss' : self.val_loss}                
            self.update_experiment_results(results)