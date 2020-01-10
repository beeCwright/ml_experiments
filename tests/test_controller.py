from controller import ExperimentController

config_path='../../demo/demo_congig.yaml'
ec = ExperimentController(config_path)

def test_get_design():
	# Delete the collection in the config
	ec.col.drop()

	# Test creating a new design

	# Test retrieveing a design



