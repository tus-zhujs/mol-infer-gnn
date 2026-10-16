from .inverter_base import *
from .ann_inverter import *
from .gnn_inverter import *

def create_inverter(inv_type, inv):
	if inv_type == "ANN":
		new_inv = ANNInverter(name=inv)
	elif inv_type == "GNN":
		new_inv = GNNInverter(name=inv)
	else:
		raise ValueError(f"Unknown learning model type: {inv_type} !")

	return new_inv
