import pytest
from ml_experiments.base import BaseConnection, BaseReader


def test_BaseConnection():
    # Check that valid experiment values work
    # Instantiate the class
    bc = BaseConnection()

    # Test the the correct prefix with experiments values works
    prefix = 'controller'
    bc.experiment = {prefix + '_host' : 'd7920-12.ccds.io', 
    prefix + '_ssl_ca_file' : './pki/rootCA.pem',
    prefix + '_ssl_certfile' : './pki/client.pem',
    prefix + '_database' : 'test_databases', 
    prefix + '_collection' : 'test_collection',
    prefix + '_port' : 27024}
    bc.establish_db_connection(prefix=prefix)

    # Test that the wrong prefix with experiment values is asserted
    wrong_prefix = 'not_controller'
    bc.experiment = {prefix + '_host' : 'd7920-12.ccds.io', 
    prefix + '_ssl_ca_file' : './pki/rootCA.pem',
    prefix + '_ssl_certfile' : './pki/client.pem',
    prefix + '_database' : 'test_databases', 
    prefix + '_collection' : 'test_collection',
    prefix + '_port' : 27024}

    with pytest.raises(AssertionError):
        bc.establish_db_connection(prefix=wrong_prefix)


def test_BaseReader():
    # Check the the config file can be read, and bundles correctly
    br = BaseReader()
    br._load_config(config='./demo/demo_config.yaml')

    br._bundle_experiment()

    # Check that controller is a header
    assert 'controller' in br.config_keys

    # Check the number of elements under the subheadings equals the length of the experiment
    num_elements = 0
    for key in br.config_keys:
        num_elements += len(br.config_dict[key])
    assert num_elements == len(br.experiment)


def test_fix_none():
    # Check that the number of 'None' and None are the same
    br = BaseReader()
    br._load_config(config='./demo/demo_config.yaml')
    br._bundle_experiment()
    br._fix_none()

    # Count the number of python Nones in the experiment
    experiment_none_count = 0
    for k,v in br.experiment.items():
        if v == None:
            experiment_none_count += 1

    # Count the number of str Nones
    config_none_count = 0
    for key in br.config_keys:
        for sub_key in list(br.config_dict[key].keys()):
            if br.config_dict[key][sub_key] == 'None':
                config_none_count += 1

    # Check that they are the same
    assert experiment_none_count == config_none_count


def test_bundle_experiments():
    br = BaseReader()
    br._load_config(config='./demo/demo_config.yaml')
    br._bundle_experiment()
    br._fix_none()
    br._bundle_hyperparameters()

    # Check that all hyperparameters are included in the experiment bounds
    assert len(br.hyperparameter_names) == len(br.bounds), 'Invalid experiment'
