import pytest
import numpy as np
import pandas as pd
import pkg_resources
from ml_experiments.manager import ExperimentNamer, ExperimentRecorder

en = ExperimentNamer()

def test_get_used_names():
    test_df = pd.DataFrame()
    en.get_used_names(test_df)
    assert en.used_names == [], 'Used names initalized incorrectly'

    test_df = pd.DataFrame({'experiment_name' : ['einstein', 'hooke'], 'random_experiment_value' : [123, 456]})
    en.get_used_names(test_df)
    assert len(en.used_names) == 2, 'Used names initalized incorrectly'


def test_get_unused_names():
    test_df = pd.DataFrame({'experiment_name' : ['einstein', 'hooke'], 'random_experiment_value' : [123, 456]})
    en.get_used_names(test_df)
    en.get_unused_names()
    assert len(en.unused_names) == len(en.name_pool) - 2


def test_get_random_unused_name():
    test_df = pd.DataFrame({'experiment_name' : ['einstein', 'hooke'], 'random_experiment_value' : [123, 456]})
    en.get_used_names(test_df)
    en.get_unused_names()
    random_name = en.get_random_unused_name()
    assert random_name not in en.unused_names
    assert len(en.unused_names) == len(en.name_pool) - 3



filepath = './demo/demo_config.yaml'
er = ExperimentRecorder(filepath)

def test_init():
    assert er.experiment['experiment_name'] in er.name_pool
    assert er.record_metrics == True, 'train_metric_name did not evaluate correctly'
    assert er.train_metric_name == er.experiment['train_metric_name']
    
def test_on_epoch_begin():
    # Test that there is no database record
    er.on_epoch_begin(epoch=1)
    with pytest.raises(AssertionError):
        assert hasattr(er, 'entry_id')
    
    # Test that there is a database record
    er.on_epoch_begin(epoch=er.start_recording)
    assert hasattr(er, 'entry_id')
    
def test_create_experiment_entry():
    er.create_experiment_entry()
    entry = er.db.col.find_one({'_id' : er.entry_id.inserted_id})
    assert entry['experiment_name'] == er.experiment['experiment_name']


def test_update_experiment_results():
    results = {'loss' : [0.6, 0.5, 0.4], 'val_loss' : [0.7, 0.6, 0.5]}
    
    # Check that the results don't exist in the database entry
    with pytest.raises(AssertionError):
        entry = er.db.col.find_one({'_id' : er.entry_id.inserted_id})
        assert 'loss' in list(entry.keys())
    
    # Insert the results
    er.update_experiment_results(results)
    entry = er.db.col.find_one({'_id' : er.entry_id.inserted_id})
    
    # Check that the results now exist in the database entry
    assert entry['loss'] == [0.6, 0.5, 0.4]
    assert entry['val_loss'] == [0.7, 0.6, 0.5]


def test_on_epoch_end():
    logs = {'loss' : 0.5, 'val_loss' : 0.6}
    er.on_epoch_end(1, logs)
    entry = er.db.col.find_one({'_id' : er.entry_id.inserted_id})
    assert entry['loss'] ==  [0.6, 0.5, 0.4]

    er.loss = [0.6, 0.5, 0.4]
    er.on_epoch_end(er.start_recording, logs)
    entry = er.db.col.find_one({'_id' : er.entry_id.inserted_id})
    assert entry['loss'] ==  [0.6, 0.5, 0.4, 0.5]