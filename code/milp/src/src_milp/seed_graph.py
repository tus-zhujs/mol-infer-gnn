from .seed_graph_2LMM_GNN import *

def create_seed_graph(sg_type, sg):
	if sg_type == "2LMM_GNN":
		new_sg = SeedGraph_2LMM_GNN(name=sg)
	else:
		raise ValueError(f"Unknown seed graph type: {sg_type} !")

	return new_sg
