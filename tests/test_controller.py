import pytest
import numpy as np
from ml_experiments.controller import ExperimentController

config_path='./demo/demo_config.yaml'
ec = ExperimentController(config_path)

def test_get_design():
    # Delete the collection in the config
    ec.col.drop()
    print(ec.col.estimated_document_count())
    assert ec.col.estimated_document_count() == 0

    # Test creating a new design
    ec._get_design()
    assert ec.col.estimated_document_count() == 1

    # Test that two designs aren't created
    ec._get_design()
    assert ec.col.estimated_document_count() == 1


def test_create_design_entry():
    # Check the warm up array is the correct shape
    assert np.shape(ec.warm_up_list) == (ec.num_warm_up, len(ec.bounds))

    # Check the document actually exists in mongo and the values are correct
    experiment = ec.col.find_one()
    assert experiment['_id'] == ec.raster_id
    assert experiment['next_iter'] == ec.next_iter
    assert experiment['num_warm_up'] == ec.num_warm_up
    assert experiment['warm_up_list'] == ec.warm_up_list

def test_tally_an_iteration():
    # Test that the tally actually increases
    cur_iter = ec.next_iter
    ec._tally_an_iteration()
    assert cur_iter + 1 == ec.next_iter
    experiment = ec.col.find_one()
    assert experiment['next_iter'] == ec.next_iter


def test_do_bayesian_optimization():
    # Check the shapes of bayesian optimzation
    # Because the inference values are sampled, a deterministic sample can't be tested
    num_hyperparameters = len(ec.bounds)
    ec.X_steps = [[0.1]*num_hyperparameters]*ec.num_warm_up
    ec.Y_steps = [[0.2]]*ec.num_warm_up
    next_trial = ec._do_bayesian_optimization()
    assert np.shape(next_trial) == (num_hyperparameters,)


def update_design():
    # Test that a specific value is actually being changed
    num_hyperparameters = len(ec.bounds)
    x_step = [0.1]*num_hyperparameters
    y_step = 0.2

    # Test that input fails without y_step being a list
    with pytest.raises(AssertionError):
        ec.update_design(x_step, y_step)

    # Test that values are 
    y_step = [0.2]
    X_steps = ec.X_steps
    Y_steps = ec.Y_steps
    ec.update_design(x_step, y_step)
    ec._get_design()
    assert ec.X_steps[-1] == x_step
    assert ec.Y_steps[-1] == y_step


def test_get_next_suggestion():

    # Set the next steps
    num_hyperparameters = len(ec.bounds)
    ec.X_steps = [[0.1]*num_hyperparameters]*ec.num_warm_up
    ec.Y_steps = [[0.2]]*ec.num_warm_up
    ec._set_val('X_steps', ec.X_steps)
    ec._set_val('Y_steps', ec.Y_steps)	

    # Check that the next trial comes off the warm up list
    ec._set_val('next_iter', 5)
    next_trial = ec.get_next_suggestion()
    assert next_trial == ec.warm_up_list[5]

    # Check that the next trial is not on the warm up list
    ec._set_val('next_iter', ec.num_warm_up + 1)
    next_trial = ec.get_next_suggestion()
    for warm_up in ec.warm_up_list:
        assert (next_trial == warm_up).all() == False


