import os, sys, datetime

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score, balanced_accuracy_score, mean_absolute_error
from sklearn.utils import class_weight
from scipy.special import softmax
# from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import numpy as np
import random

from src import chemicalgraph, dataset, network, constants

import itertools, json, time, optuna

from typing import Callable, Iterable, Optional

_num_workers = 4
_pin_memory = True
_persistent_workers = True
_batch_size = 256

RANDOM_SEED = 4242
RANDOM_SEED_BASE = 10000

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std=1.0):
	mae = 0
	ema.assign(model)
	model.eval()
	with torch.no_grad():
		with torch.amp.autocast('cuda', dtype=amp_dtype):
			for data in loader:
				data = data.to(device, non_blocking=True)
				output = model(data, ft_dataset_batch) if ft_dataset_used else model(data)
				mae += (output.float() - data.y.float()).abs().sum().item()
				y_true = data.y.detach().to(torch.float32).cpu().numpy()
				y_pred = output.detach().to(torch.float32).cpu().numpy()
	ema.resume(model)
	model.train()
	model_mae = mae / len(loader.dataset) * y_std
	model_r2 = r2_score(y_true, y_pred)

	return model_mae, model_r2, y_true, y_pred

def learn_GNN_selfmade_trainval(
	train_dataset, val_dataset, ft_dataset, params, 
	extra_x_length=None, batch_size=_batch_size, num_epoch=30,
	random_seed=RANDOM_SEED, verbose=True, save=False, 
	output_prefix="GNN",
	resume_filename=None,
	y_mean=0.0,
	y_std=1.0,
	callback: Optional[Callable[[int, float, float, float], None]] = None,
	eval_epochs: Optional[Iterable[int]] = None
):
	set_seed(random_seed)

	tmp_result_filename = os.path.join(".", "tmp_result.txt")
	with open(tmp_result_filename, 'w') as f:
		f.write("Start !!!\n")

	timestamp = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
	log_prefix = os.path.join(".", "log", f"{output_prefix}_{timestamp}_pre")

	model_name = params['model'] if 'model' in params.keys() else 'GNN_selfmade'
	hidden_size = params['hidden_size']
	p_size = params['p_size']
	num_layers = params['num_layers']
	ANN_channels = params['ANN_channels']
	weight_decay = params['weight_decay']
	learning_rate = params['learning_rate']

	num_output = ANN_channels[-1]

	num_atom_features = params['num_atom_features']

	# metric_name = params['metric']

	# if metric_name == 'r2':
	metric = r2_score

	mae_train_summary = []
	mae_val_summary = []
	timestamp_summary = []
	best_val_loss = None

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=_num_workers, pin_memory=_pin_memory, persistent_workers=_persistent_workers)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=_num_workers, pin_memory=_pin_memory, persistent_workers=_persistent_workers)

	ft_dataloader = DataLoader(ft_dataset, batch_size=len(ft_dataset))
	for ft_dataset_batch in ft_dataloader:
		break

	ft_dataset_used = False

	# if model_name == "GNN_int":
	#     model = network.GNN_int(
	#         in_channels=num_atom_features,
	#         hidden_channels=hidden_size,
	#         num_layers=num_layers,
	#         p_channels=p_size,
	#         ANN_channels=ANN_channels,
	#         num_edge_types=3,
	#         extra_x_length=extra_x_length
	#     )
	# elif model_name == "GNN_int_no_edge_type":
	#     model = network.GNN_int_no_edge_type(
	#         in_channels=num_atom_features,
	#         hidden_channels=hidden_size,
	#         num_layers=num_layers,
	#         p_channels=p_size,
	#         ANN_channels=ANN_channels,
	#         extra_x_length=extra_x_length
	#     )
	
	if model_name == "GNN_2L":
		model = network.GNN_2L(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=64,
			num_layers_FT=6,
			p_channels_FT=128,
			ANN_channels_FT=(128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			num_edge_types=3,
			extra_x_length=extra_x_length,
			dropout=0.2
		)
		ft_dataset_used = True
	elif model_name == "GNN_2L_no_edge_type":
		model = network.GNN_2L_no_edge_type(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=256,
			num_layers_FT=16,
			p_channels_FT=256,
			ANN_channels_FT=(256, 128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			extra_x_length=extra_x_length,
			dropout=0.2
		)
		ft_dataset_used = True
	elif model_name == "GNN_2L_no_edge_type_GraphSAGE":
		model = network.GNN_2L_no_edge_type_GraphSAGE(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=256,
			num_layers_FT=16,
			p_channels_FT=256,
			ANN_channels_FT=(256, 128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			extra_x_length=extra_x_length,
			dropout=0.2
		)
		ft_dataset_used = True
	# model = network.GraphSAGE(
	#     in_channels=train_dataset.num_node_features,
	#     hidden_channels=hidden_size,
	#     num_layers=num_layers,
	#     out_channels=1
	# )

	# model = network.GraphATConv(
	#     in_channels=train_dataset.num_node_features,
	#     hidden_channels=hidden_size,
	#     num_layers=num_layers,
	#     out_channels=1
	# )

	config_name = os.path.join(".", "model", f"{output_prefix}_pre_config.txt")

	with open(config_name, 'w') as f:
		f.write(f"in_channels_FT, {model.in_channels_FT}\n")
		f.write(f"hidden_channels_FT, {model.hidden_channels_FT}\n")
		f.write(f"num_layers_FT, {model.num_layers_FT}\n")
		f.write(f"p_channels_FT, {model.p_channels_FT}\n")
		ANN_channels_tmp = [str(ttt) for ttt in model.ANN_channels_FT]
		ANN_str = ", ".join(ANN_channels_tmp)
		f.write(f"ANN_channels_FT, {ANN_str}\n")
		f.write(f"in_channels_IN, {num_atom_features}\n")
		f.write(f"hidden_channels_IN, {hidden_size}\n")
		f.write(f"num_layers_IN, {num_layers}\n")
		f.write(f"p_channels_IN, {p_size}\n")
		ANN_channels_tmp = [str(ttt) for ttt in ANN_channels]
		ANN_str = ", ".join(ANN_channels_tmp) 
		f.write(f"ANN_channels_IN, {ANN_str}\n")
		f.write(f"y_mean, {y_mean}\n")
		f.write(f"y_std, {y_std}\n")

	device = torch.device(constants.get_default_device())

	model.to(device)
	ft_dataset_batch.to(device, non_blocking=True)
	print("Number of model parameters: ", count_parameters(model))
	print("Number of model parameters: ", count_parameters(model), file=sys.stderr, flush=True)

	if save is True:
		output_model_name = os.path.join(".", "model", f"{output_prefix}.pth")
		print(f"The model will be saved as {output_model_name}")
		print(f"The model will be saved as {output_model_name}", file=sys.stderr, flush=True)
	else:
		output_model_name = None
	early_stopping = network.EarlyStopping(patience=20, verbose=False, path=output_model_name)

	if resume_filename is not None:
		checkpoint = torch.load(resume_filename)
		model.load_state_dict(checkpoint)
	if output_model_name is not None:
		torch.save(model.state_dict(), output_model_name)

	total_refresh_time = 1 # tmp value

	print(f"Epoch, loss_train, score_train, score_test, time")
	print(f"Epoch, loss_train, score_train, score_test, time", file=sys.stderr, flush=True)

	for _ in range(total_refresh_time):

		if output_model_name is not None:
			checkpoint = torch.load(output_model_name)
			model.load_state_dict(checkpoint)

		ema = network.EMA(model, decay=0.999)

		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
		
		warmup_epochs = 1
		base_lr = optimizer.param_groups[0]['lr']
		warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
		# cosine_sched = CosineAnnealingLR(optimizer, T_max=num_epoch-warmup_epochs, eta_min=base_lr*1e-2)
		exp_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
		scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, exp_sched], milestones=[warmup_epochs])
		
		# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
		# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

		time_epoch_start = time.time()
		scaler = torch.amp.GradScaler('cuda')

		for epoch in range(num_epoch):
			running_loss = 0
			model.train()

			for index, batch in enumerate(train_loader):
				batch = batch.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)

				with torch.amp.autocast('cuda', dtype=amp_dtype):
					out = model(batch, ft_dataset_batch) if ft_dataset_used else model(batch)
					loss = F.l1_loss(out, batch.y)
				running_loss += loss.detach().item() * len(batch)
				scaler.scale(loss).backward()
				# loss.backward()
				scaler.unscale_(optimizer)
				clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
				# optimizer.step()
				scaler.step(optimizer)
				scaler.update()

				# curr_epoch = epoch + float(index) / (len(train_dataset) / batch_size)
				# scheduler_warmup.step(curr_epoch)

				ema(model)
			
			scheduler.step()

			running_loss_mae = running_loss / len(train_dataset) * y_std
			val_loss_mae, val_loss_r2, val_true, val_pred = test(model, val_loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std)
			# train_loss_mae, train_loss_r2 = test(model, train_all_loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std)
			mae_train_summary.append(running_loss_mae)
			mae_val_summary.append(val_loss_mae)
			timestamp_summary.append(epoch + 1)

			if best_val_loss is None or val_loss_mae <= best_val_loss:
				best_val_loss = val_loss_mae
				best_val_loss_r2 = val_loss_r2
				best_train_loss = running_loss_mae

				if output_model_name is not None:
					torch.save(model.state_dict(), output_model_name)

				output_str = f'Epoch: {epoch+1:04d}, Train MAE: {running_loss_mae:.7f}, Val MAE: {val_loss_mae:.7f}, Val R2: {val_loss_r2:.7f}, Time: {time.time() - time_epoch_start:.7f}.'
				print(output_str)
				print("\r"+output_str, end='', file=sys.stderr, flush=True)
				with open(tmp_result_filename, 'a') as f:
					f.write(output_str + "\n")

			if eval_epochs is not None and (epoch+1) in set(eval_epochs):
				try:
					if callback is not None:
						callback(epoch + 1, val_loss_mae, val_loss_r2, running_loss_mae)
				except Exception as e:
					raise

			early_stopping(val_loss_mae, model, epoch)
			if early_stopping.early_stop:
				print("Early stopping!")
				print("\nEarly stopping!", file=sys.stderr, flush=True)
				break

	with open(f"{log_prefix}_timestamp.txt", 'a') as f:
		f.write(f"Epoch, mae_train, mae_test\n")
		for _epoch, _mae_train, _mae_test in zip(timestamp_summary, mae_train_summary, mae_val_summary):
			f.write(f"{_epoch}, {_mae_train}, {_mae_test}\n")

	with open(f"{log_prefix}_valset.txt", 'w') as f:
		f.write(f"Val:\n")
		f.write(f"val_true, val_pred\n")
		for i in range(num_output):
			for (v_t, v_p) in zip(list(val_true), list(val_pred)):
				f.write(f"{v_t[i]}, \t {v_p[i]}\n")
			f.write("\n")

	print(f"Log file saved in {log_prefix}_timestamp.txt!!!")
	print(f"\nLog file saved in {log_prefix}_timestamp.txt!!!", file=sys.stderr, flush=True)

	return best_val_loss, best_val_loss_r2, best_train_loss

def learn_GNN_selfmade_trainvaltest(
	train_dataset, val_dataset, test_dataset, ft_dataset, params, 
	extra_x_length=None, batch_size=_batch_size, num_epoch=300,
	random_seed=RANDOM_SEED, verbose=True, save=False, 
	output_prefix="GNN",
	resume_filename=None,
	y_mean=0.0,
	y_std=1.0
):
	set_seed(random_seed)

	tmp_result_filename = os.path.join(".", "tmp_result.txt")
	with open(tmp_result_filename, 'w') as f:
		f.write("Start !!!\n")

	timestamp = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
	log_prefix = os.path.join(".", "log", f"{output_prefix}_{timestamp}")

	model_name = params['model'] if 'model' in params.keys() else 'GNN_selfmade'
	hidden_size = params['hidden_size']
	p_size = params['p_size']
	num_layers = params['num_layers']
	ANN_channels = params['ANN_channels']
	weight_decay = params['weight_decay']
	learning_rate = params['learning_rate']

	num_output = ANN_channels[-1]

	num_atom_features = params['num_atom_features']

	# metric_name = params['metric']

	# if metric_name == 'r2':
	metric = r2_score

	mae_train_summary = []
	mae_val_summary = []
	timestamp_summary = []
	best_val_loss = None

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=_num_workers, pin_memory=_pin_memory, persistent_workers=_persistent_workers)
	val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=_num_workers, pin_memory=_pin_memory, persistent_workers=_persistent_workers)
	test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=_num_workers, pin_memory=_pin_memory, persistent_workers=_persistent_workers)

	# train_all_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

	ft_dataloader = DataLoader(ft_dataset, batch_size=len(ft_dataset))
	for ft_dataset_batch in ft_dataloader:
		break

	ft_dataset_used = False

	# if model_name == "GNN_int":
	#     model = network.GNN_int(
	#         in_channels=num_atom_features,
	#         hidden_channels=hidden_size,
	#         num_layers=num_layers,
	#         p_channels=p_size,
	#         ANN_channels=ANN_channels,
	#         num_edge_types=3,
	#         extra_x_length=extra_x_length
	#     )
	# elif model_name == "GNN_int_no_edge_type":
	#     model = network.GNN_int_no_edge_type(
	#         in_channels=num_atom_features,
	#         hidden_channels=hidden_size,
	#         num_layers=num_layers,
	#         p_channels=p_size,
	#         ANN_channels=ANN_channels,
	#         extra_x_length=extra_x_length
	#     )
	
	if model_name == "GNN_2L":
		model = network.GNN_2L(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=64,
			num_layers_FT=6,
			p_channels_FT=128,
			ANN_channels_FT=(128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			num_edge_types=3,
			extra_x_length=extra_x_length,
			dropout=0.2
		)
		ft_dataset_used = True
	elif model_name == "GNN_2L_no_edge_type":
		model = network.GNN_2L_no_edge_type(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=256,
			num_layers_FT=16,
			p_channels_FT=256,
			ANN_channels_FT=(256, 128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			extra_x_length=extra_x_length
			# dropout=0.2
		)
		ft_dataset_used = True
	elif model_name == "GNN_2L_no_edge_type_GraphSAGE":
		model = network.GNN_2L_no_edge_type_GraphSAGE(
			in_channels_FT=ft_dataset.num_node_features,
			hidden_channels_FT=256,
			num_layers_FT=16,
			p_channels_FT=256,
			ANN_channels_FT=(256, 128, 64, 32, 16, 8),
			in_channels_IN=num_atom_features,
			hidden_channels_IN=hidden_size,
			num_layers_IN=num_layers,
			p_channels_IN=p_size,
			ANN_channels_IN=ANN_channels,
			extra_x_length=extra_x_length,
			dropout=0.2
		)
		ft_dataset_used = True
	# model = network.GraphSAGE(
	#     in_channels=train_dataset.num_node_features,
	#     hidden_channels=hidden_size,
	#     num_layers=num_layers,
	#     out_channels=1
	# )

	# model = network.GraphATConv(
	#     in_channels=train_dataset.num_node_features,
	#     hidden_channels=hidden_size,
	#     num_layers=num_layers,
	#     out_channels=1
	# )

	config_name = os.path.join(".", "model", f"{output_prefix}_config.txt")

	with open(config_name, 'w') as f:
		f.write(f"in_channels_FT, {model.in_channels_FT}\n")
		f.write(f"hidden_channels_FT, {model.hidden_channels_FT}\n")
		f.write(f"num_layers_FT, {model.num_layers_FT}\n")
		f.write(f"p_channels_FT, {model.p_channels_FT}\n")
		ANN_channels_tmp = [str(ttt) for ttt in model.ANN_channels_FT]
		ANN_str = ", ".join(ANN_channels_tmp)
		f.write(f"ANN_channels_FT, {ANN_str}\n")
		f.write(f"in_channels_IN, {num_atom_features}\n")
		f.write(f"hidden_channels_IN, {hidden_size}\n")
		f.write(f"num_layers_IN, {num_layers}\n")
		f.write(f"p_channels_IN, {p_size}\n")
		ANN_channels_tmp = [str(ttt) for ttt in ANN_channels]
		ANN_str = ", ".join(ANN_channels_tmp) 
		f.write(f"ANN_channels_IN, {ANN_str}\n")
		f.write(f"y_mean, {y_mean}\n")
		f.write(f"y_std, {y_std}\n")

	device = torch.device(constants.get_default_device())

	model.to(device)
	ft_dataset_batch.to(device, non_blocking=True)
	print("Number of model parameters: ", count_parameters(model))
	print("Number of model parameters: ", count_parameters(model), file=sys.stderr, flush=True)

	if save is True:
		output_model_name = os.path.join(".", "model", f"{output_prefix}.pth")
		print(f"The model will be saved as {output_model_name}")
		print(f"The model will be saved as {output_model_name}", file=sys.stderr, flush=True)
	else:
		output_model_name = None
	early_stopping = network.EarlyStopping(patience=100, verbose=False, path=output_model_name)

	if resume_filename is not None:
		checkpoint = torch.load(resume_filename)
		model.load_state_dict(checkpoint)
	if output_model_name is not None:
		torch.save(model.state_dict(), output_model_name)
		
	total_refresh_time = 1 # tmp value

	print(f"Epoch, loss_train, score_train, score_test, time")
	print(f"Epoch, loss_train, score_train, score_test, time", file=sys.stderr, flush=True)

	for _ in range(total_refresh_time):

		if output_model_name is not None:
			checkpoint = torch.load(output_model_name)
			model.load_state_dict(checkpoint)

		ema = network.EMA(model, decay=0.999)

		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
		
		warmup_epochs = 1
		base_lr = optimizer.param_groups[0]['lr']
		warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
		# cosine_sched = CosineAnnealingLR(optimizer, T_max=num_epoch-warmup_epochs, eta_min=base_lr*1e-2)
		exp_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
		scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, exp_sched], milestones=[warmup_epochs])
		
		# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
		# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

		time_epoch_start = time.time()
		scaler = torch.amp.GradScaler('cuda')

		for epoch in range(num_epoch):
			running_loss = 0
			model.train()

			for index, batch in enumerate(train_loader):
				batch = batch.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)

				with torch.amp.autocast('cuda', dtype=amp_dtype):
					out = model(batch, ft_dataset_batch) if ft_dataset_used else model(batch)
					loss = F.l1_loss(out, batch.y)
				running_loss += loss.detach().item() * len(batch)
				scaler.scale(loss).backward()
				# loss.backward()
				scaler.unscale_(optimizer)
				clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
				# optimizer.step()
				scaler.step(optimizer)
				scaler.update()

				# curr_epoch = epoch + float(index) / (len(train_dataset) / batch_size)
				# scheduler_warmup.step(curr_epoch)

				ema(model)
				
			scheduler.step()

			running_loss_mae = running_loss / len(train_dataset)
			val_loss_mae, val_loss_r2, val_true, val_pred = test(model, val_loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std)
			# train_loss_mae, train_loss_r2 = test(model, train_all_loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std)
			mae_train_summary.append(running_loss_mae)
			mae_val_summary.append(val_loss_mae)
			timestamp_summary.append(epoch + 1)

			if best_val_loss is None or val_loss_mae <= best_val_loss:
				test_loss_mae, test_loss_r2, test_true, test_pred = test(model, test_loader, ema, device, ft_dataset_used, ft_dataset_batch, y_std)
				best_val_loss = val_loss_mae
				if output_model_name is not None:
					torch.save(model.state_dict(), output_model_name)

				# with open(f"{log_prefix}_testset.txt", 'w') as f:
				#     f.write("val_true, val_pred\n")
				#     for _true, _pred in zip(test_true, test_pred):
				#         f.write(f"{_true}, {_pred}\n")
				output_str = f'Epoch: {epoch+1:04d}, Train MAE: {running_loss_mae * y_std:.7f}, Val MAE: {val_loss_mae:.7f}, Val R2: {val_loss_r2:.7f}, Test MAE: {test_loss_mae:.7f}, Test R2: {test_loss_r2:.7f}, Time: {time.time() - time_epoch_start:.7f}.'
				print(output_str)
				print("\r"+output_str, end='', file=sys.stderr, flush=True)
				with open(tmp_result_filename, 'a') as f:
					f.write(output_str + "\n")
					
			early_stopping(val_loss_mae, model, epoch)
			if early_stopping.early_stop:
				print("Early stopping!")
				print("\nEarly stopping!", file=sys.stderr, flush=True)
				break

	# print(f"{metric_name}_train: {r2_train_score_tmp}\n{metric_name}_test: {r2_test_score_tmp}\n")

	with open(f"{log_prefix}_timestamp.txt", 'a') as f:
		f.write(f"Epoch, mae_train, mae_test\n")
		for _epoch, _mae_train, _mae_test in zip(timestamp_summary, mae_train_summary, mae_val_summary):
			f.write(f"{_epoch}, {_mae_train}, {_mae_test}\n")

	with open(f"{log_prefix}_valset.txt", 'w') as f:
		f.write(f"Val:\n")
		f.write(f"val_true, val_pred\n")
		for i in range(num_output):
			for (v_t, v_p) in zip(list(val_true), list(val_pred)):
				f.write(f"{v_t[i]}, \t {v_p[i]}\n")
			f.write("\n")

	with open(f"{log_prefix}_testset.txt", 'w') as f:
		f.write(f"Test:\n")
		f.write(f"test_true, test_pred\n")
		for i in range(num_output):
			for (t_t, t_p) in zip(list(test_true), list(test_pred)):
				f.write(f"{t_t[i]}, \t {t_p[i]}\n")
			f.write("\n")


	print(f"Log file saved in {log_prefix}_timestamp.txt!!!")
	print(f"\nLog file saved in {log_prefix}_timestamp.txt!!!", file=sys.stderr, flush=True)

	print('Best Validation MAE:', best_val_loss)
	print('Testing MAE:', test_loss_mae)
	print('Testing R2:', test_loss_r2)
	print('Best Validation MAE:', best_val_loss, file=sys.stderr, flush=True)
	print('Testing MAE:', test_loss_mae, file=sys.stderr, flush=True)
	print('Testing R2:', test_loss_r2, file=sys.stderr, flush=True)

	return best_val_loss, test_loss_mae, test_loss_r2

# def hyper_param_tuning_trainval(train_dataset, val_dataset, ft_dataset, extra_x_length, search_params, y_mean=0.0, y_std=1.0):

#     best_val_score = None
#     best_val_score_r2 = None
#     best_train_score = None
#     best_params = None

#     keys = list(search_params)
#     for values in itertools.product(*map(search_params.get, keys)):
#         params = dict(zip(keys, values))

#         print(f"params: {json.dumps(params)}")
#         print(f"params: {json.dumps(params)}", file=sys.stderr, flush=True)

#         time_start_org = time.time()

#         # for split_seed in range(1, Times_pre+1):
#         best_val_loss, best_val_loss_r2, best_train_loss = \
#             learn_GNN_selfmade_trainval(train_dataset, val_dataset, ft_dataset, params, 
#                 extra_x_length=extra_x_length, random_seed=RANDOM_SEED_BASE, num_epoch=30, y_mean=y_mean, y_std=y_std)

#         print(f"Best Validation MAE: {best_val_loss}")
#         print(f"Validation R2: {best_val_loss_r2}")
#         print(f"Train MAE: {best_train_loss}")
#         print(f"Time: {time.time() - time_start_org}s\n")
#         print(f"Best Validation MAE: {best_val_loss}", file=sys.stderr, flush=True)
#         print(f"Validation R2: {best_val_loss_r2}", file=sys.stderr, flush=True)
#         print(f"Train MAE: {best_train_loss}", file=sys.stderr, flush=True)
#         print(f"Time: {time.time() - time_start_org}s\n", file=sys.stderr, flush=True)

#         if best_val_score is None or best_val_score > best_val_loss:
#             best_val_score = best_val_loss
#             best_val_score_r2 = best_val_loss_r2
#             best_train_score = best_train_loss
#             best_params = params.copy()

#     return best_params, best_val_score, best_val_score_r2, best_train_score

def hyper_param_tuning_trainval(
	train_dataset, val_dataset, ft_dataset, extra_x_length, search_params, y_mean=0.0, y_std=1.0,
	n_trials=20,       
	stages=(20, 40, 60),   #
	reduction_factor=3, 
	grace_epoch=8,        
	primary_metric="mae"   
):
	base_params = {}
	for k, v in search_params.items():
		if k in ("learning_rate", "weight_decay", "metric"):
			continue  # lr/wd will be determined by Optuna
		base_params[k] = v[0] if isinstance(v, (list, tuple)) else v

	lr_range = (search_params["learning_rate"][0], search_params["learning_rate"][1])
	wd_range = (search_params["weight_decay"][0], search_params["weight_decay"][1])

	def objective(trial: optuna.Trial):
		lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
		wd_zero = trial.suggest_categorical("wd_zero", [True, False])
		weight_decay = 0.0 if wd_zero else trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)

		print(f"trial#{trial.number} start: lr={lr:.6g}, wd={weight_decay:.6g} (wd_zero={wd_zero})")
		print(f"trial#{trial.number} start: lr={lr:.6g}, wd={weight_decay:.6g} (wd_zero={wd_zero})",
			  file=sys.stderr, flush=True)

		params = dict(base_params)
		params["learning_rate"] = lr
		params["weight_decay"] = weight_decay

		best_val_mae_overall = None
		best_val_r2_overall = None
		best_train_mae_overall = None
		eval_points = tuple(stages)

		def cb(ep, val_mae, val_r2, train_mae):
			nonlocal best_val_mae_overall, best_val_r2_overall, best_train_mae_overall
			if best_val_mae_overall is None or val_mae < best_val_mae_overall:
				best_val_mae_overall = val_mae
				best_val_r2_overall = val_r2
				best_train_mae_overall = train_mae

			report_value = -val_r2 if primary_metric.lower() == "r2" else val_mae
			trial.report(report_value, step=ep)

			print(f"  stage@{ep}ep -> val_MAE={val_mae:.6f}, val_R2={val_r2:.6f}, "
				  f"train_MAE={train_mae:.6f}")
			print(f"  stage@{ep}ep -> val_MAE={val_mae:.6f}, val_R2={val_r2:.6f}, "
				  f"train_MAE={train_mae:.6f}",
				  file=sys.stderr, flush=True)

			if ep >= grace_epoch and trial.should_prune():
				raise optuna.TrialPruned()

		val_mae, val_r2, train_mae = learn_GNN_selfmade_trainval(
			train_dataset, val_dataset, ft_dataset, params,
			extra_x_length=extra_x_length,
			num_epoch=max(stages),
			y_mean=y_mean, y_std=y_std,
			verbose=False,          
			callback=cb,           
			eval_epochs=eval_points 
		)

		if best_val_mae_overall is None:
			best_val_mae_overall = val_mae
			best_val_r2_overall  = val_r2
			best_train_mae_overall = train_mae

		return (-best_val_r2_overall) if primary_metric.lower() == "r2" else best_val_mae_overall

	pruner = optuna.pruners.HyperbandPruner(
		min_resource=grace_epoch,                 
		max_resource=max(stages),                 
		reduction_factor=reduction_factor      
	)

	direction = "minimize"  

	study = optuna.create_study(direction=direction, pruner=pruner, sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
	study.optimize(objective, n_trials=n_trials)

	best = study.best_trial
	best_lr = best.params["learning_rate"]
	best_wd = 0.0 if best.params.get("wd_zero", False) else best.params["weight_decay"]

	best_params = dict(base_params)
	best_params["learning_rate"] = best_lr
	best_params["weight_decay"] = best_wd

	final_val_mae, final_val_r2, final_train_mae = learn_GNN_selfmade_trainval(
		train_dataset, val_dataset, ft_dataset, best_params,
		extra_x_length=extra_x_length,
		num_epoch=max(stages),
		y_mean=y_mean, y_std=y_std,
		verbose=False
	)

	print("\n=== Optuna best ===")
	print(json.dumps({"learning_rate": best_lr, "weight_decay": best_wd}, ensure_ascii=False, indent=2))
	print(f"Val MAE: {final_val_mae:.6f}, Val R2: {final_val_r2:.6f}, Train MAE: {final_train_mae:.6f}")
	print("\n=== Optuna best ===", file=sys.stderr, flush=True)
	print(json.dumps({"learning_rate": best_lr, "weight_decay": best_wd}, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
	print(f"Val MAE: {final_val_mae:.6f}, Val R2: {final_val_r2:.6f}, Train MAE: {final_train_mae:.6f}", file=sys.stderr, flush=True)

	return best_params, final_val_mae, final_val_r2, final_train_mae

def eval_GNN(train_dataset, val_dataset, test_dataset, ft_dataset, extra_x_length, best_params, num_epoch=300, output_prefix="GNN", resume_filename=None, y_mean=0.0, y_std=1.0, random_seed=RANDOM_SEED):
	# train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

	print(f"\nEVALUATION:")
	print(f"params: {json.dumps(best_params)}")
	print(f"\nEVALUATION:", file=sys.stderr, flush=True)
	print(f"params: {json.dumps(best_params)}", file=sys.stderr, flush=True)

	time_start_org = time.time()

	best_val_loss, test_loss_mae, test_loss_r2 = learn_GNN_selfmade_trainvaltest(train_dataset, val_dataset, test_dataset, ft_dataset, best_params, 
		extra_x_length=extra_x_length, random_seed=random_seed, save=True, num_epoch=num_epoch, output_prefix=output_prefix, resume_filename=resume_filename, y_mean=y_mean, y_std=y_std)

	print(f"Time: {time.time() - time_start_org}s\n")
	print(f"Time: {time.time() - time_start_org}s\n", file=sys.stderr, flush=True)

	return best_val_loss, test_loss_mae, test_loss_r2

