import os, json, sys, argparse, time, datetime, warnings
import torch, torch_geometric

# warnings.simplefilter('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_filename = os.path.join(BASE_DIR, "result_file.txt")

from src import GNN, chemicalgraph, dataset, network, constants

import warnings
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module=r"torch_geometric\.data\.collate"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.optim\.lr_scheduler"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch_geometric\.data\.in_memory_dataset"
)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("input_sdf", help='input sdf file')
	parser.add_argument("input_value", help='input value file')
	# parser.add_argument("output_pre", help='input output directory for results of preliminary experiments')
	# parser.add_argument("output_eval", help='input output file for results of evaluation experiments')
	parser.add_argument('-e', '--exfv', nargs='+', type=str)
	parser.add_argument('-c', '--clsf', nargs='+', type=str)
	# parser.add_argument('-s', '--standardization', nargs='*', type=str)   
	parser.add_argument('-m', '--metric', nargs='+', type=str)  

	parser.add_argument('-l', '--layer_num', nargs='+', type=int)
	parser.add_argument('-hn', '--hidden_node', nargs='+', type=int)

	parser.add_argument('-val', '--val', nargs='+', type=str) 

	parser.add_argument('-o', '--output_prefix', nargs='+', type=str)

	parser.add_argument('-r', '--resume', nargs='+', type=str)

	args = parser.parse_args()

	return(args) 

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_y(data, mean, std):
	data.y = (data.y - mean) / std
	return data

def main(args):
	sdf_filename = args.input_sdf
	value_filename = args.input_value
	extra_fv_filename = None
	extra_x_length = None

	time_start = time.time()

	if args.exfv is not None:
		extra_fv_filename = args.exfv[0]

	prop_name = "QM9"

	print(f"Run for {prop_name} dataset, property: {args.val}")
	print(f"Run for {prop_name} dataset, property: {args.val}", file=sys.stderr, flush=True)

	# read sdf file
	if extra_fv_filename is not None:
		cg_list = chemicalgraph.construct_from_SDF_file(sdf_filename, extra_fv_used=True, check_qm9=True)
		extra_x_length = cg_list.read_extra_fv_file(extra_fv_filename)
	else:
		cg_list = chemicalgraph.construct_from_SDF_file(sdf_filename, use_rdkit=False, check_qm9=True)

	HAR2EV = 27.2113825435
	KCALMOL2EV = 0.04336414

	conversion = {
		"mu": 1.0, "alpha": 1.0, "homo": HAR2EV,
		"lumo": HAR2EV, "gap": HAR2EV, "r2": 1.0, "zpve": HAR2EV, "u0": HAR2EV,
		"u298": HAR2EV, "h298": HAR2EV, "g298": HAR2EV, "cv": 1.0, "u0_atom": KCALMOL2EV,
		"u298_atom": KCALMOL2EV, "h298_atom": KCALMOL2EV, "g298_atom": KCALMOL2EV
	}

	cg_list.read_value_file_csv(value_filename, args.val, conversion)

	# cg_list_without_H = chemicalgraph.create_H_suppresed_chemical_graph(cg_list)

	cg_list.get_fringe_tree()

	cg_dataset = dataset.ChemicalGraph_Datasets(root=os.path.join(BASE_DIR, 'data'))
	cg_dataset.construct_from_ChemicalGraph_list(cg_list, FT_use=True)

	ft_dataset = dataset.FringeTree_Datasets(root=os.path.join(BASE_DIR, 'data'))
	ft_dataset.construct_from_FringeTree_list(cg_list.fringe_tree_list)

	num_cg_atom_features = cg_dataset.data.x.shape[1]
	num_cg_edge_features = cg_dataset.data.edge_attr.shape[1]
	num_ft_atom_features = ft_dataset.data.x.shape[1]
	num_ft_edge_features = ft_dataset.data.edge_attr.shape[1]

	print("Dataset reading completed!")
	print(f"cg_len: {len(cg_dataset)}, num_atom_features: {num_cg_atom_features}, num_edge_features: {num_cg_edge_features}")
	print(f"ft_len: {len(ft_dataset)}, num_atom_features: {num_ft_atom_features}, num_edge_features: {num_ft_edge_features}")
	print(f"Learning start, device={constants.get_default_device()}", file=sys.stderr, flush=True)

	print("Dataset reading completed!", file=sys.stderr, flush=True)
	print(f"cg_len: {len(cg_dataset)}, num_atom_features: {num_cg_atom_features}, num_edge_features: {num_cg_edge_features}", file=sys.stderr, flush=True)
	print(f"ft_len: {len(ft_dataset)}, num_atom_features: {num_ft_atom_features}, num_edge_features: {num_ft_edge_features}", file=sys.stderr, flush=True)
	print(f"Learning start, device={constants.get_default_device()}", file=sys.stderr, flush=True)

	# SEARCH_PARAMS = {
	#     # 'model': ['GNN_int_no_edge_type'],
	#     'model': ['GNN_2L_no_edge_type'],
	#     'num_atom_features': [num_cg_atom_features],
	#     # 'model': ['GNN_2L'],
	#     'hidden_size': [32],
	#     'num_layers': [1],
	#     # 'ANN_channels': [(16, 1), (32, 1), (64, 1), (16, 16, 1), (32, 16, 1), (64, 32, 1), (64, 32, 16, 1)],
	#     'ANN_channels': [(32, 1)],
	#     'weight_decay': [1e-4],
	#     'metric': ['r2'] # r2, aucroc, bacc
	# }

	SEARCH_PARAMS = {
		'model': ['GNN_2L_no_edge_type'],
		'num_atom_features': [num_cg_atom_features],
		'hidden_size': [int(args.hidden_node[0])],
		'p_size': [32],
		'num_layers': [int(args.layer_num[0])],
		'ANN_channels': [(16, len(args.val))],
		'learning_rate': [5e-5, 1e-3],
		'weight_decay': [1e-7, 5e-4],
		'metric': ['r2'] # r2, aucroc, bacc
	}

	if args.metric is not None:
		SEARCH_PARAMS['metric'] = args.metric

	if args.clsf is not None:
		SEARCH_PARAMS['clsf'] = [n_class]

	if args.resume is not None:
		resume_filename = args.resume[0]
	else:
		resume_filename = None

	# train_size = int(len(cg_dataset) * 0.8)
	# val_size = int(len(cg_dataset) * 0.1)
	# test_size = len(cg_dataset) - train_size - val_size

	train_size = 110000
	val_size = 10000
	test_size = len(cg_dataset) - train_size - val_size

	generator = torch.Generator().manual_seed(42)

	print("Hypertuning start!")
	print("Hypertuning start!", file=sys.stderr, flush=True)

	train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(cg_dataset, [train_size, val_size, test_size], generator=generator)
	y_train = torch.cat([data.y for data in train_dataset], dim=0)
	y_mean, y_std = y_train.mean(), y_train.std()
	train_dataset = [normalize_y(d, y_mean, y_std) for d in train_dataset]
	val_dataset = [normalize_y(d, y_mean, y_std) for d in val_dataset]
	test_dataset = [normalize_y(d, y_mean, y_std) for d in test_dataset]

	best_params, best_val_score, best_val_score_r2, best_train_score = \
		GNN.hyper_param_tuning_trainval(train_dataset, val_dataset, ft_dataset, extra_x_length, search_params=SEARCH_PARAMS, y_mean=y_mean, y_std=y_std)

	# print(f"Best_params: {json.dumps(best_params)}")
	# print(f"Best Validation MAE: {best_val_score}, Validation R2: {best_val_score_r2}, Train MAE: {best_train_score}")
	# print(f"Best_params: {json.dumps(best_params)}", file=sys.stderr, flush=True)
	# print(f"Best Validation MAE: {best_val_score}, Validation R2: {best_val_score_r2}, Train MAE: {best_train_score}", file=sys.stderr, flush=True)

	output_prefix = args.output_prefix[0]

	# best_params = {
	#         'model': 'GNN_2L_no_edge_type',
	#         'num_atom_features': num_cg_atom_features,
	#         'hidden_size': 16,
	#         'p_size': 32,
	#         'num_layers': 3,
	#         'ANN_channels': (16,len(args.val)),
	#         'weight_decay': 0,
	#         'metric': 'r2' # r2, aucroc, bacc
	#     }

	best_val_loss, test_loss_mae, test_loss_r2 = \
		GNN.eval_GNN(train_dataset, val_dataset, test_dataset, ft_dataset, extra_x_length, best_params, num_epoch=3000, output_prefix=output_prefix, resume_filename=resume_filename, y_mean=y_mean, y_std=y_std)

	# for name, param in model.named_parameters():
	#     if param.requires_grad:
	#         print(name, param.data)     


if __name__ == "__main__":
	print("torch:", torch.__version__)
	print("cuda:", torch.cuda.is_available(), "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "| bf16:", torch.cuda.is_bf16_supported())
	print("device count:", torch.cuda.device_count())

	print("torch:", torch.__version__, file=sys.stderr, flush=True)
	print("cuda:", torch.cuda.is_available(), "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "| bf16:", torch.cuda.is_bf16_supported(), file=sys.stderr, flush=True)
	print("device count:", torch.cuda.device_count(), file=sys.stderr, flush=True)


	main(get_args())





