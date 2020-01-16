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

# def test_init():
#     assert 