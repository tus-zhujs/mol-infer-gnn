from .inverter_base import *
import pulp
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, ClassVar
import pandas as pd
import numpy as np
from src.utils import _read_clean_lines, _pop_int, _pop_ints, _pop_floats, _pop_float, run_with_retry
from .ann_inverter import ANNInverter

import torch
from torch_geometric.loader import DataLoader
from .src_gnn import chemicalgraph, network, dataset

def _cRT_to_RT(psi):
	RT = chemicalgraph.RootedTree()
	RT.CID = psi.index
	for (i, v) in enumerate(psi.vertex):
		atom = chemicalgraph.Atom()
		atom.index = i
		atom.atom_type = v[0].rstrip('1234567890')
		atom.atom_name = v[0]
		atom.status = True
		atom.chg = psi.chg[i]

		RT.vertex_list[i] = atom

	n = len(psi.vertex)
	for i in range(n):
		for j in range(i + 1, n):
			if psi.beta[i][j] != 0:
				RT.parent[j] = i
				v1, v2, mul = i, j, psi.beta[i][j]
				RT.vertex_list[v1].add_neighbour(v2, mul, RT.vertex_list[v2].atom_type)
				RT.vertex_list[v2].add_neighbour(v1, mul, RT.vertex_list[v1].atom_type)

	RT.root = 0

	return RT

@dataclass(frozen=False, kw_only=True)
class GNNInverter(InverterBase):
	### An inverter class for GNN
	name: str
	_type: ClassVar[Literal["INV_GNN"]] = "INV_GNN"

	def __post_init__(self, name="gnn"):
		self.output = dict()
		self.index_set = dict() # store the relation between the indices of fringe trees in fv files and in seed graph files, index_set[index_in_fv] = index_in_seed_graph
		self.x_hat = dict() # store the normalized feature vector value
		self.std_eps = 1e-5

		self.theta_cv_tmp = dict()
		self.delta_cv_tmp = dict()
		self.theta_cv = dict()
		self.theta_C_tmp = dict()
		self.delta_C_tmp = dict()
		self.theta_C = dict()
		self.theta_T_tmp = dict()
		self.delta_T_tmp = dict()
		self.theta_T = dict()
		self.theta_F_tmp = dict()
		self.delta_F_tmp = dict()
		self.theta_F = dict()
		self.theta_C_minus = dict()
		self.theta_C_plus = dict()
		self.theta_T_minus = dict()
		self.theta_T_plus = dict()
		self.theta_F_minus = dict()
		self.theta_F_plus = dict()
		self.theta_CT_T = dict()
		self.theta_TC_T = dict()
		self.theta_CT_C = dict()
		self.theta_TC_C = dict()
		self.theta_TF_F = dict()
		self.theta_CF_F = dict()
		self.theta_CF_C = dict()
		self.theta_TF_T = dict()

	def build_ft_dataset(self, GNN, set_F):
		fringe_tree_list = list()
		ft_index_map = dict()
		for i, psi in enumerate(set_F):
			rt = _cRT_to_RT(psi)
			fringe_tree_list.append(rt)
			ft_index_map[psi.index] = i
		ft_dataset = dataset.FringeTree_Datasets(root='./data')
		ft_dataset.construct_from_FringeTree_list(fringe_tree_list)

		ft_dataloader = DataLoader(ft_dataset, batch_size=len(ft_dataset))
		for ft_dataset_batch in ft_dataloader:
			break

		ft_dataset_batch.to(self.device)
		GNN.eval()
		self.ft_embedding = GNN.GNN_FT(ft_dataset_batch).detach().numpy()
		self.ft_embedding = {psi.index: self.ft_embedding[ft_index_map[psi.index], :] for psi in set_F}

	def _get_weights(self, learned_weights):
		self.weight_w = dict()
		self.weight_bias = dict()
		self.weight_w_cv = dict()

		p_cv = self.p_cv
		l_max = self.num_layers
		z_max = self.num_node_features
		h_size = self.hidden_layer_size

		for l in range(0, l_max):
			for z_p in range(1, (z_max if l == 0 else h_size) + 1):
				for z in range(1, h_size + 1):
					self.weight_w[(0, l, z_p, z)] = learned_weights[f'GNN_2L_int.convs.{l}.lin_l_self.weight'][z-1, z_p-1].item()
					self.weight_w[(1, l, z_p, z)] = learned_weights[f'GNN_2L_int.convs.{l}.lin_l.weight'][z-1, z_p-1].item()
			for z in range(1, h_size + 1):
				self.weight_bias[(0, z, l)] = learned_weights[f'GNN_2L_int.convs.{l}.bias'][z-1].item()
		for p in range(1, p_cv + 1):
			for z in range(1, h_size + 1):
				self.weight_w_cv[(z, p)] = learned_weights['GNN_2L_int.convs_to_p.weight'][p-1, z-1].item()

		self.ann_weight_tensor = list()
		self.ann_bias_matrix = list()

		self.ann_bias_matrix.append([0] * self.ann_layers[0])

		for i in range(len(self.ann_layers) - 1):
			tmp_array = learned_weights[f'GNN_2L_int.Linears.{i}.weight'].numpy().T
			self.ann_weight_tensor.append(np.array(tmp_array))
			tmp_array = learned_weights[f'GNN_2L_int.Linears.{i}.bias'].numpy()
			self.ann_bias_matrix.append(np.array(tmp_array))

	def read_GNN_config(self, config_file):
		self.params = dict()
		with open(config_file, 'r') as f:
			line = f.readline()
			while line:
				line_all = [l.rstrip(',\n') for l in line.split(',')]
				param_name = line_all[0]
				if 'ANN' in param_name:
					arr = tuple([int(n) for n in line_all[1:]])
				elif param_name == "y_mean" or param_name == "y_std":
					arr = float(line_all[1])
				else:
					arr = int(line_all[1])
				self.params[param_name] = arr
				line = f.readline()

		if 'extra_x_length' not in self.params:
			self.params['extra_x_length'] = None

	def _linear_bounds(self, weights, input_bounds, bias=0.0):
		"""Bounds of sum_i weights[i] * x_i + bias, where x_i in [L_i, U_i]."""
		lb = float(bias)
		ub = float(bias)

		for a, (L, U) in zip(weights, input_bounds):
			a = float(a)
			L = float(L)
			U = float(U)
			if a >= 0:
				lb += a * L
				ub += a * U
			else:
				lb += a * U
				ub += a * L

		return lb, ub


	def _leaky_relu_bounds(self, L, U, kappa=0.1):
		"""Bounds of LeakyReLU(x) = max(x, kappa*x), x in [L, U]."""
		L = float(L)
		U = float(U)

		if U <= 0:
			return kappa * L, kappa * U
		if L >= 0:
			return L, U
		return kappa * L, U


	def _sum_bounds(self, bounds_list):
		"""Bounds of the sum of all listed interval terms."""
		if not bounds_list:
			return 0.0, 0.0
		return (
			sum(L for L, _ in bounds_list),
			sum(U for _, U in bounds_list),
		)


	def _sum_at_most_k_bounds(self, bounds_list, k):
		"""
		Safe bounds for the sum of at most k active terms.
		Each term can also be inactive, hence 0 is always available.
		"""
		if not bounds_list or k <= 0:
			return 0.0, 0.0

		k = min(k, len(bounds_list))

		lows = sorted(min(0.0, float(L)) for L, _ in bounds_list)
		ups = sorted((max(0.0, float(U)) for _, U in bounds_list), reverse=True)

		return sum(lows[:k]), sum(ups[:k])


	def compute_nodewise_GNN_bounds(
		self,
		sg,
		*,
		kappa=0.1,
		safety=1.05,
		T_msg_active_ub=2,
		F_msg_active_ub=2,
		cap_C_by_valence=False,
		verbose=True,
	):
		"""
		Compute nodewise interval bounds and Big-M values for the GNN part.

		This assumes the GNN update has the form

			preact_v = W_self h_v + sum_{u in N(v)} W_msg h_u + bias,

		where the self term is handled separately.

		Returned dictionaries use keys:
			(node_type, node_index, layer, feature)
		where node_type is "C", "T", or "F".

		Important:
			layer = 0 means input node feature.
			layer = 1,...,l_max means hidden layers after LeakyReLU.
			preact_bounds are stored for layer = 1,...,l_max.
		"""
		l_max = self.num_layers
		z_max = self.num_node_features
		h_size = self.hidden_layer_size
		p_cv = self.p_cv

		w = self.weight_w
		bias = self.weight_bias
		w_cv = self.weight_w_cv
		ft_embedding = self.ft_embedding

		t_C = sg.t_C
		t_C_tilde = sg.t_C_tilde
		t_T = sg.t_T
		t_F = sg.t_F

		val = sg.val
		max_val = max(val.values())

		# ------------------------------------------------------------
		# Helper: node sets
		# ------------------------------------------------------------
		node_ranges = {
			"C": range(1, t_C + 1),
			"T": range(1, t_T + 1),
			"F": range(1, t_F + 1),
		}

		fringe_candidates = {
			"C": sg.F_C,
			"T": sg.F_T,
			"F": sg.F_F,
		}

		# ------------------------------------------------------------
		# Layer 0 bounds
		# ------------------------------------------------------------
		theta_bounds = {}
		preact_bounds = {}

		def _embedding_values_for_node(node_type, i, j):
			vals = []
			for psi in fringe_candidates[node_type][i]:
				code = sg.Code_F[psi]
				vals.append(float(ft_embedding[code][j - 8]))

			# Fallback: should rarely be needed, but avoids empty min/max.
			if not vals:
				vals = [float(emb[j - 8]) for emb in ft_embedding.values()]

			return vals

		for node_type, nodes in node_ranges.items():
			for i in nodes:
				# Atom type indicators.
				theta_bounds[(node_type, i, 0, 1)] = (0.0, 1.0)
				theta_bounds[(node_type, i, 0, 2)] = (0.0, 1.0)
				theta_bounds[(node_type, i, 0, 3)] = (0.0, 1.0)

				# degree + hyddeg, valence - eledeg, hyddeg, eledeg.
				# These are deliberately safe bounds.
				theta_bounds[(node_type, i, 0, 4)] = (0.0, float(max_val))
				theta_bounds[(node_type, i, 0, 5)] = (0.0, float(max_val))
				theta_bounds[(node_type, i, 0, 6)] = (0.0, float(max_val))
				theta_bounds[(node_type, i, 0, 7)] = (0.0, float(max_val))

				# Fringe-tree embedding features.
				for j in range(8, z_max + 1):
					vals = _embedding_values_for_node(node_type, i, j)
					theta_bounds[(node_type, i, 0, j)] = (
						float(min(vals)),
						float(max(vals)),
					)

		# ------------------------------------------------------------
		# Nodewise message term counts / active upper bounds
		# ------------------------------------------------------------
		msg_active_ub = {"C": {}, "T": {}, "F": {}}

		for i in range(1, t_C + 1):
			# This is the number of message expressions appearing in the C update.
			# For core C nodes, theta_CF_F contributes one possible F-message.
			c_expr_count = (
				len(sg.I_a_minus[i])
				+ len(sg.I_a_plus[i])
				+ len(sg.I_b_minus[i])
				+ len(sg.I_b_plus[i])
				+ (1 if i <= t_C_tilde else 0)
			)

			if cap_C_by_valence:
				local_valence_ub = max(val[a] for a in sg.Lambda_star[i])
				c_expr_count = min(c_expr_count, local_valence_ub)

			msg_active_ub["C"][i] = c_expr_count

		for i in range(1, t_T + 1):
			msg_active_ub["T"][i] = T_msg_active_ub

		for i in range(1, t_F + 1):
			msg_active_ub["F"][i] = F_msg_active_ub

		# ------------------------------------------------------------
		# Propagate bounds layer by layer
		# ------------------------------------------------------------
		msg_affine_bounds = {}
		self_affine_bounds = {}

		for l in range(0, l_max):
			input_dim = z_max if l == 0 else h_size

			# Bounds for W_self h_v and W_msg h_v for each possible source node.
			for node_type, nodes in node_ranges.items():
				for i in nodes:
					input_bounds = [
						theta_bounds[(node_type, i, l, z_p)]
						for z_p in range(1, input_dim + 1)
					]

					for z in range(1, h_size + 1):
						self_weights = [
							w[(0, l, z_p, z)]
							for z_p in range(1, input_dim + 1)
						]
						msg_weights = [
							w[(1, l, z_p, z)]
							for z_p in range(1, input_dim + 1)
						]

						self_affine_bounds[(node_type, i, l, z)] = self._linear_bounds(
							self_weights,
							input_bounds,
							bias=0.0,
						)
						msg_affine_bounds[(node_type, i, l, z)] = self._linear_bounds(
							msg_weights,
							input_bounds,
							bias=0.0,
						)

			# Worst message bounds by source type.
			# Used when the exact source node is selected by chi variables.
			worst_msg_by_type = {}
			for node_type, nodes in node_ranges.items():
				for z in range(1, h_size + 1):
					candidates = [
						msg_affine_bounds[(node_type, i, l, z)]
						for i in nodes
					]
					if candidates:
						worst_msg_by_type[(node_type, z)] = (
							min(L for L, _ in candidates),
							max(U for _, U in candidates),
						)
					else:
						worst_msg_by_type[(node_type, z)] = (0.0, 0.0)

			for z in range(1, h_size + 1):
				# -------------------------
				# C nodes
				# -------------------------
				for i in range(1, t_C + 1):
					self_L, self_U = self_affine_bounds[("C", i, l, z)]

					msg_candidates = []

					# C-C messages with exact C source.
					for k in sg.I_a_minus[i]:
						src = sg.tail_C[k]
						msg_candidates.append(msg_affine_bounds[("C", src, l, z)])

					for k in sg.I_a_plus[i]:
						src = sg.head_C[k]
						msg_candidates.append(msg_affine_bounds[("C", src, l, z)])

					# T messages selected by chi_T. Source T is variable, so use worst T bound.
					for _ in sg.I_b_minus[i]:
						msg_candidates.append(worst_msg_by_type[("T", z)])

					for _ in sg.I_b_plus[i]:
						msg_candidates.append(worst_msg_by_type[("T", z)])

					# F message selected by chi_F for core C nodes.
					if i <= t_C_tilde:
						msg_candidates.append(worst_msg_by_type[("F", z)])

					if cap_C_by_valence:
						msg_L, msg_U = self._sum_at_most_k_bounds(
							msg_candidates,
							msg_active_ub["C"][i],
						)
					else:
						msg_L, msg_U = self._sum_bounds(msg_candidates)

					pre_L = bias[(0, z, l)] + self_L + msg_L
					pre_U = bias[(0, z, l)] + self_U + msg_U

					preact_bounds[("C", i, l + 1, z)] = (pre_L, pre_U)
					theta_bounds[("C", i, l + 1, z)] = self._leaky_relu_bounds(
						pre_L,
						pre_U,
						kappa=kappa,
					)

				# -------------------------
				# T nodes
				# -------------------------
				for i in range(1, t_T + 1):
					self_L, self_U = self_affine_bounds[("T", i, l, z)]

					msg_candidates = []

					# T-chain neighbors.
					if i >= 2:
						msg_candidates.append(msg_affine_bounds[("T", i - 1, l, z)])
					if i <= t_T - 1:
						msg_candidates.append(msg_affine_bounds[("T", i + 1, l, z)])

					# C attachment messages. Tail/head C are chosen by chi_T.
					msg_candidates.append(worst_msg_by_type[("C", z)])
					msg_candidates.append(worst_msg_by_type[("C", z)])

					# F attachment message.
					msg_candidates.append(worst_msg_by_type[("F", z)])

					msg_L, msg_U = self._sum_at_most_k_bounds(
						msg_candidates,
						msg_active_ub["T"][i],
					)

					pre_L = bias[(0, z, l)] + self_L + msg_L
					pre_U = bias[(0, z, l)] + self_U + msg_U

					preact_bounds[("T", i, l + 1, z)] = (pre_L, pre_U)
					theta_bounds[("T", i, l + 1, z)] = self._leaky_relu_bounds(
						pre_L,
						pre_U,
						kappa=kappa,
					)

				# -------------------------
				# F nodes
				# -------------------------
				for i in range(1, t_F + 1):
					self_L, self_U = self_affine_bounds[("F", i, l, z)]

					msg_candidates = []

					# F-chain neighbors.
					if i >= 2:
						msg_candidates.append(msg_affine_bounds[("F", i - 1, l, z)])
					if i <= t_F - 1:
						msg_candidates.append(msg_affine_bounds[("F", i + 1, l, z)])

					# C and T attachment messages.
					msg_candidates.append(worst_msg_by_type[("C", z)])
					msg_candidates.append(worst_msg_by_type[("T", z)])

					msg_L, msg_U = self._sum_at_most_k_bounds(
						msg_candidates,
						msg_active_ub["F"][i],
					)

					pre_L = bias[(0, z, l)] + self_L + msg_L
					pre_U = bias[(0, z, l)] + self_U + msg_U

					preact_bounds[("F", i, l + 1, z)] = (pre_L, pre_U)
					theta_bounds[("F", i, l + 1, z)] = self._leaky_relu_bounds(
						pre_L,
						pre_U,
						kappa=kappa,
					)

		# ------------------------------------------------------------
		# Readout bounds
		# ------------------------------------------------------------
		cv_bounds = {}

		for p in range(1, p_cv + 1):
			weights = []
			bounds = []

			for node_type, nodes in node_ranges.items():
				for i in nodes:
					for z in range(1, h_size + 1):
						weights.append(w_cv[(z, p)])
						bounds.append(theta_bounds[(node_type, i, l_max, z)])

			cv_bounds[p] = self._linear_bounds(weights, bounds, bias=0.0)

		cv_act_bounds = {}
		for p, (L, U) in cv_bounds.items():
			cv_act_bounds[p] = self._leaky_relu_bounds(L, U, kappa=kappa)

		# ------------------------------------------------------------
		# Convert bounds to Big-M values
		# ------------------------------------------------------------
		def _M_from_bounds(bounds):
			L, U = bounds
			return safety * max(abs(float(L)), abs(float(U)), 1e-9)

		M_theta = {
			key: _M_from_bounds(bounds)
			for key, bounds in theta_bounds.items()
		}

		M_preact = {
			key: _M_from_bounds(bounds)
			for key, bounds in preact_bounds.items()
		}

		M_msg = {
			key: _M_from_bounds(bounds)
			for key, bounds in msg_affine_bounds.items()
		}

		M_cv = {
			p: _M_from_bounds(bounds)
			for p, bounds in cv_bounds.items()
		}

		# Layer-wise fallback values, useful while replacing old M[l].
		M_layer = {}
		for l in range(0, l_max + 1):
			vals = []

			vals.extend(
				M
				for (node_type, i, ll, z), M in M_theta.items()
				if ll == l
			)

			vals.extend(
				M
				for (node_type, i, ll, z), M in M_preact.items()
				if ll == l
			)

			M_layer[l] = max(vals) if vals else 1.0

		M_readout_layer = max(M_cv.values()) if M_cv else 1.0

		# Message layer fallback: bounds for W_msg h_v at layer l.
		M_msg_layer = {}
		for l in range(0, l_max):
			vals = [
				M
				for (node_type, i, ll, z), M in M_msg.items()
				if ll == l
			]
			M_msg_layer[l] = max(vals) if vals else 1.0

		self.gnn_bounds = {
			"theta_bounds": theta_bounds,
			"preact_bounds": preact_bounds,
			"cv_bounds": cv_bounds,
			"cv_act_bounds": cv_act_bounds,
			"msg_affine_bounds": msg_affine_bounds,
			"self_affine_bounds": self_affine_bounds,
			"M_theta": M_theta,
			"M_preact": M_preact,
			"M_msg": M_msg,
			"M_cv": M_cv,
			"M_layer": M_layer,
			"M_readout_layer": M_readout_layer,
			"M_msg_layer": M_msg_layer,
			"msg_active_ub": msg_active_ub,
		}

		if verbose:
			print("[GNN Big-M] msg_active_ub C:", msg_active_ub["C"])
			print("[GNN Big-M] msg_active_ub T max:", max(msg_active_ub["T"].values(), default=0))
			print("[GNN Big-M] msg_active_ub F max:", max(msg_active_ub["F"].values(), default=0))
			print("[GNN Big-M] M_conv_layer:", M_layer)
			print("[GNN Big-M] M_msg_layer:", M_msg_layer)
			print("[GNN Big-M] M_readout_layer:", M_readout_layer)
			print("[GNN Big-M] M_cv max:", max(M_cv.values(), default=0))

		return self.gnn_bounds

	def set_up_inverter(self, invl):
		self.sdf_filename = invl.sdf_filename
		self.fv_gen_path = None
		self._use_quadratic = None

		model_filename = invl.model_filename
		config_filename = invl.config_filename

		try:
			self.read_GNN_config(config_filename)
		except:
			raise ValueError(f"Error: Failed to read the inverter {self.name}!")

		self.device = torch.device('cpu')
		self.GNN = network.GNN_2L_no_edge_type(
				in_channels_FT=self.params['in_channels_FT'],
				hidden_channels_FT=self.params['hidden_channels_FT'],
				num_layers_FT=self.params['num_layers_FT'],
				p_channels_FT=self.params['p_channels_FT'],
				ANN_channels_FT=self.params['ANN_channels_FT'],
				in_channels_IN=self.params['in_channels_IN'],
				hidden_channels_IN=self.params['hidden_channels_IN'],
				num_layers_IN=self.params['num_layers_IN'],
				p_channels_IN=self.params['p_channels_IN'],
				ANN_channels_IN=self.params['ANN_channels_IN'],
				extra_x_length=self.params['extra_x_length'],
				device=self.device
			)

		self.learned_weights = torch.load(
			model_filename, map_location=self.device, weights_only=True
		)
		self.GNN.load_state_dict(self.learned_weights)

		self.p_cv = self.params['p_channels_IN']
		self.num_layers = self.params['num_layers_IN']
		self.num_node_features = self.params['in_channels_IN'] + self.params['ANN_channels_FT'][-1]
		self.hidden_layer_size = self.params['hidden_channels_IN']
		self.ann_layers = (self.p_cv,) + self.params['ANN_channels_IN']
		self.y_mean = self.params['y_mean']
		self.y_std = self.params['y_std']

		self._get_weights(self.learned_weights)

		# self.inv_ann = ANNInverter(name=f"{self.name}_ANN")
		# self.inv_ann.set_up_inverter_for_GNN(self.ann_weight_tensor, self.ann_bias_matrix, self.ann_layers)

	def set_up_variables(self, asgml, sg):
		t_C = sg.t_C
		t_C_tilde = sg.t_C_tilde
		t_T = sg.t_T
		t_F = sg.t_F
		k_C = sg.k_C
		k_C_tilde = sg.k_C_tilde
		m_C = sg.m_C

		p_cv = self.p_cv
		l_max = self.num_layers
		z_max = self.num_node_features
		h_size = self.hidden_layer_size

		# Define theta_cv
		self.theta_cv_tmp[asgml.ind] = {p: pulp.LpVariable(f"GNN_theta_cv_tmp({p})_ASGM{asgml.ind}") for p in range(1, p_cv + 1)}
		self.delta_cv_tmp[asgml.ind] = {p: pulp.LpVariable(f"GNN_delta_cv_tmp({p})_ASGM{asgml.ind}", 0, 1, cat=pulp.LpBinary) for p in range(1, p_cv + 1)}
		self.theta_cv[asgml.ind] = {p: pulp.LpVariable(f"GNN_theta_cv({p})_ASGM{asgml.ind}") for p in range(1, p_cv + 1)}

		# Define theta_C
		self.theta_C_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_C_tmp({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_C + 1)
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.delta_C_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_delta_C_tmp({i},{z},{ll})_ASGM{asgml.ind}", 0, 1, cat=pulp.LpBinary) for i in range(1, t_C + 1)
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.theta_C[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_C({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_C + 1)
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}

		# Define theta_T
		self.theta_T_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_T_tmp({i},{z},{ll})_ASGM{asgml.ind}") for i in range(0, t_T + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.delta_T_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_delta_T_tmp({i},{z},{ll})_ASGM{asgml.ind}", 0, 1, cat=pulp.LpBinary) for i in range(0, t_T + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.theta_T[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_T({i},{z},{ll})_ASGM{asgml.ind}") for i in range(0, t_T + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}

		# Define theta_F
		self.theta_F_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_F_tmp({i},{z},{ll})_ASGM{asgml.ind}") for i in range(0, t_F + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.delta_F_tmp[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_delta_F_tmp({i},{z},{ll})_ASGM{asgml.ind}", 0, 1, cat=pulp.LpBinary) for i in range(0, t_F + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}
		self.theta_F[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_F({i},{z},{ll})_ASGM{asgml.ind}") for i in range(0, t_F + 2)  # for convenience
					for ll in range(0, l_max + 1) for z in range(1, (z_max if ll == 0 else h_size) + 1)}

		# Define theta_C_minus, theta_C_plus
		self.theta_C_minus[asgml.ind] = {(k, z, ll): pulp.LpVariable(f"GNN_theta_C_minus({k},{z},{ll})_ASGM{asgml.ind}") for k in range(k_C_tilde + 1, m_C + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_C_plus[asgml.ind] = {(k, z, ll): pulp.LpVariable(f"GNN_theta_C_plus({k},{z},{ll})_ASGM{asgml.ind}") for k in range(k_C_tilde + 1, m_C + 1) # instead of [1, k_C], it should be [k_C_tilde + 1, m_C]
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_T_minus, theta_T_plus
		self.theta_T_minus[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_T_minus({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_T + 2)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_T_plus[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_T_plus({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_T + 2)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_F_minus, theta_F_plus
		self.theta_F_minus[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_F_minus({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_F + 2)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_F_plus[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_F_plus({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_F + 2)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_CT_T, theta_TC_T
		self.theta_CT_T[asgml.ind] = {(k, z, ll): pulp.LpVariable(f"GNN_theta_CT_T({k},{z},{ll})_ASGM{asgml.ind}") for k in range(1, k_C + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_TC_T[asgml.ind] = {(k, z, ll): pulp.LpVariable(f"GNN_theta_TC_T({k},{z},{ll})_ASGM{asgml.ind}") for k in range(1, k_C + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_CT_C, theta_TC_C, theta_TF_F
		self.theta_CT_C[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_CT_C({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_T + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)} # why R+
		self.theta_TC_C[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_TC_C({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_T + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_TF_F[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_TF_F({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_T + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_CF_F
		self.theta_CF_F[asgml.ind] = {(c, z, ll): pulp.LpVariable(f"GNN_theta_CF_F({c},{z},{ll})_ASGM{asgml.ind}") for c in range(1, t_C_tilde + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		# Define theta_CF_C, theta_TF_T
		self.theta_CF_C[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_CF_C({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_F + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}
		self.theta_TF_T[asgml.ind] = {(i, z, ll): pulp.LpVariable(f"GNN_theta_TF_T({i},{z},{ll})_ASGM{asgml.ind}") for i in range(1, t_F + 1)
					for ll in range(0, l_max) for z in range(1, h_size + 1)}

		self.inv_ann = ANNInverter(name=f"{self.name}_ANN")
		ANN_input_bounds = {
			p: self.gnn_bounds["cv_act_bounds"][p]
			for p in range(1, self.p_cv + 1)
		}
		self.inv_ann.set_up_inverter_for_GNN(ANN_input_bounds, self.ann_weight_tensor, self.ann_bias_matrix, self.ann_layers)
		self.inv_ann.set_up_variables(asgml)
		self.output[asgml.ind] = self.inv_ann.output[asgml.ind]

	def add_constraints_graph_convolution(self,
		# Model
		MILP, asgml,
		# Constants
		t_C, t_C_tilde, t_T, t_F, k_C, k_C_tilde, m_C, val, Code_F, Lambda_int, I_a_minus, I_a_plus, I_b_minus, I_b_plus, F_C, F_T, F_F, head_C, tail_C,
		# Binary Variables
		delta_alpha_C, delta_alpha_T, delta_alpha_F, delta_fr_C, delta_fr_T, delta_fr_F, delta_beta_C, delta_beta_T, delta_beta_F,
		delta_beta_plus, delta_beta_minus, delta_beta_in, e_C, e_T, e_F, delta_chi_T, delta_chi_F, chi_T, chi_F, v_T, v_F,
		# Integer Variables
		alpha_C, alpha_T, alpha_F, deg_C, deg_T, deg_F, eledeg_C, eledeg_T, eledeg_F, hyddeg_C, hyddeg_T, hyddeg_F,
		# General Variables
		theta_cv, theta_cv_tmp, delta_cv_tmp, theta_C, theta_T, theta_F, theta_C_tmp, theta_T_tmp, theta_F_tmp,
		delta_C_tmp, delta_T_tmp, delta_F_tmp, theta_C_minus, theta_C_plus, theta_T_minus, theta_T_plus, theta_F_minus, theta_F_plus,
		theta_CT_T, theta_TC_T, theta_CT_C, theta_TC_C, theta_TF_F, theta_CF_F, theta_CF_C, theta_TF_T
	):
		p_cv = self.p_cv
		l_max = self.num_layers
		z_max = self.num_node_features
		h_size = self.hidden_layer_size

		if not hasattr(self, "gnn_bounds"):
		    raise RuntimeError(
		        "Call compute_nodewise_GNN_bounds(sg) before add_constraints_graph_convolution()."
		    )

		bounds = self.gnn_bounds

		M_conv_layer = bounds.get("M_conv_layer", bounds.get("M_layer"))
		M_msg_layer = bounds["M_msg_layer"]

		M_preact = bounds["M_preact"]
		M_theta = bounds["M_theta"]
		M_msg = bounds["M_msg"]
		M_cv = bounds["M_cv"]

		def m_pre(node_type, i, l_next, z):
		    return M_preact.get((node_type, i, l_next, z), M_conv_layer[l_next])

		def m_theta(node_type, i, l, z):
		    return M_theta.get((node_type, i, l, z), M_conv_layer[l])

		def m_msg(node_type, i, l, z):
		    return M_msg.get((node_type, i, l, z), M_msg_layer[l])

		kappa = 0.1

		ft_embedding = self.ft_embedding

		w, bias, w_cv = self.weight_w, self.weight_bias, self.weight_w_cv

		# -------- Constraint (69) --------
		for i in range(1, t_C + 1):
			MILP += theta_C[(i, 1, 0)] == delta_alpha_C[(i, "C4")], f"MILP-2LM-M-GC-(65)-C-1-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 2, 0)] == delta_alpha_C[(i, "O2")], f"MILP-2LM-M-GC-(65)-C-2-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 3, 0)] == delta_alpha_C[(i, "N3")], f"MILP-2LM-M-GC-(65)-C-3-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 4, 0)] == deg_C[i] + hyddeg_C[i], f"MILP-2LM-M-GC-(65)-C-4-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 5, 0)] == pulp.lpSum(val[a] * delta_alpha_C[(i, a)] for a in Lambda_int) - eledeg_C[i], f"MILP-2LM-M-GC-(65)-C-5-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 6, 0)] == hyddeg_C[i], f"MILP-2LM-M-GC-(65)-C-6-{i}_ASGM{asgml.ind}"
			MILP += theta_C[(i, 7, 0)] == eledeg_C[i], f"MILP-2LM-M-GC-(65)-C-7-{i}_ASGM{asgml.ind}"

			for j in range(8, z_max + 1):
				MILP += theta_C[(i, j, 0)] == pulp.lpSum(ft_embedding[Code_F[psi]][j - 8] * delta_fr_C[(i, Code_F[psi])] for psi in F_C[i]), f"MILP-2LM-M-GC-(65)-C-{j}-{i}_ASGM{asgml.ind}"

		for i in range(1, t_T + 1):
			MILP += theta_T[(i, 1, 0)] == delta_alpha_T[(i, "C4")], f"MILP-2LM-M-GC-(65)-T-1-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 2, 0)] == delta_alpha_T[(i, "O2")], f"MILP-2LM-M-GC-(65)-T-2-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 3, 0)] == delta_alpha_T[(i, "N3")], f"MILP-2LM-M-GC-(65)-T-3-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 4, 0)] == deg_T[i] + hyddeg_T[i], f"MILP-2LM-M-GC-(65)-T-4-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 5, 0)] == pulp.lpSum(val[a] * delta_alpha_T[(i, a)] for a in Lambda_int) - eledeg_T[i], f"MILP-2LM-M-GC-(65)-T-5-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 6, 0)] == hyddeg_T[i], f"MILP-2LM-M-GC-(65)-T-6-{i}_ASGM{asgml.ind}"
			MILP += theta_T[(i, 7, 0)] == eledeg_T[i], f"MILP-2LM-M-GC-(65)-T-7-{i}_ASGM{asgml.ind}"

			for j in range(8, z_max + 1):
				MILP += theta_T[(i, j, 0)] == pulp.lpSum(ft_embedding[Code_F[psi]][j - 8] * delta_fr_T[(i, Code_F[psi])] for psi in F_T[i]), f"MILP-2LM-M-GC-(65)-T-{j}-{i}_ASGM{asgml.ind}"

		for i in range(1, t_F + 1):
			MILP += theta_F[(i, 1, 0)] == delta_alpha_F[(i, "C4")], f"MILP-2LM-M-GC-(65)-F-1-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 2, 0)] == delta_alpha_F[(i, "O2")], f"MILP-2LM-M-GC-(65)-F-2-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 3, 0)] == delta_alpha_F[(i, "N3")], f"MILP-2LM-M-GC-(65)-F-3-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 4, 0)] == deg_F[i] + hyddeg_F[i], f"MILP-2LM-M-GC-(65)-F-4-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 5, 0)] == pulp.lpSum(val[a] * delta_alpha_F[(i, a)] for a in Lambda_int) - eledeg_F[i], f"MILP-2LM-M-GC-(65)-F-5-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 6, 0)] == hyddeg_F[i], f"MILP-2LM-M-GC-(65)-F-6-{i}_ASGM{asgml.ind}"
			MILP += theta_F[(i, 7, 0)] == eledeg_F[i], f"MILP-2LM-M-GC-(65)-F-7-{i}_ASGM{asgml.ind}"

			for j in range(8, z_max + 1):
				MILP += theta_F[(i, j, 0)] == pulp.lpSum(ft_embedding[Code_F[psi]][j - 8] * delta_fr_F[(i, Code_F[psi])] for psi in F_F[i]), f"MILP-2LM-M-GC-(65)-F-{j}-{i}_ASGM{asgml.ind}"

		# -------- Constraint (70) --------
		for z in range(1, h_size + 1):  # missing
			for l in range(0, (l_max - 1) + 1):  # missing
				for i in range(1, t_C_tilde + 1):
					Mpre = m_pre("C", i, l + 1, z)
					Mth = m_theta("C", i, l + 1, z)
					MILP += pulp.lpSum(w[(0, l, z_p, z)] * theta_C[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							pulp.lpSum(theta_C_minus[(k, z, l)] for k in I_a_minus[i]) + \
							pulp.lpSum(theta_C_plus[(k, z, l)] for k in I_a_plus[i]) + \
							pulp.lpSum(theta_CT_T[(k, z, l)] for k in I_b_plus[i]) + \
							pulp.lpSum(theta_TC_T[(k, z, l)] for k in I_b_minus[i]) + \
							theta_CF_F[(i, z, l)] + bias[(0, z, l)] == theta_C_tmp[(i, z, l + 1)], f"MILP-2LM-M-GC-(66)-{z}-{l}-{i}-1_ASGM{asgml.ind}"

					MILP += theta_C[(i, z, l + 1)] - kappa * theta_C_tmp[(i, z, l + 1)] >= -Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - kappa * theta_C_tmp[(i, z, l + 1)] <= Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - theta_C_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - theta_C_tmp[(i, z, l + 1)] <= Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_C_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-3-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C_tmp[(i, z, l + 1)] <= Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-3-{z}-{l}-{i}-2_ASGM{asgml.ind}"

					MILP += theta_C[(i, z, l + 1)] <= Mth, f"MILP-2LM-M-GC-(66)-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] >= -Mth, f"MILP-2LM-M-GC-(66)-{z}-{l}-{i}-3_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (71) --------
		for z in range(1, h_size + 1):
			for l in range(0, (l_max - 1) + 1):
				for i in range(t_C_tilde + 1, t_C + 1):
					Mpre = m_pre("C", i, l + 1, z)
					Mth = m_theta("C", i, l + 1, z)
					MILP += pulp.lpSum(w[(0, l, z_p, z)] * theta_C[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							pulp.lpSum(theta_C_minus[(k, z, l)] for k in I_a_minus[i]) + \
							pulp.lpSum(theta_C_plus[(k, z, l)] for k in I_a_plus[i]) + \
							pulp.lpSum(theta_CT_T[(k, z, l)] for k in I_b_plus[i]) + \
							pulp.lpSum(theta_TC_T[(k, z, l)] for k in I_b_minus[i]) + \
							bias[(0, z, l)] == theta_C_tmp[(i, z, l + 1)], f"MILP-2LM-M-GC-(67)-{z}-{l}-{i}-1_ASGM{asgml.ind}"

					MILP += theta_C[(i, z, l + 1)] - kappa * theta_C_tmp[(i, z, l + 1)] >= -Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - kappa * theta_C_tmp[(i, z, l + 1)] <= Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - theta_C_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C[(i, z, l + 1)] - theta_C_tmp[(i, z, l + 1)] <= Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_C_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_C_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-C-3-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_C_tmp[(i, z, l + 1)] <= Mpre * delta_C_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-3-{z}-{l}-{i}-2_ASGM{asgml.ind}"

					MILP += theta_C[(i, z, l + 1)] <= Mth, f"MILP-2LM-M-GC-(67)-{z}-{l}-{i}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_C[(i, z, l + 1)] >= -Mth, f"MILP-2LM-M-GC-(67)-{z}-{l}-{i}-3_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (72) --------
		for z in range(1, h_size + 1):
			for l in range(0, (l_max - 1) + 1):
				for i in range(1, t_T + 1):
					Mpre = m_pre("T", i, l + 1, z)
					Mth = m_theta("T", i, l + 1, z)
					MILP += theta_T_tmp[(i, z, l + 1)] - (pulp.lpSum(w[(0, l, z_p, z)] * theta_T[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							theta_T_minus[(i, z, l)] + \
							theta_T_plus[(i, z, l)] + \
							theta_CT_C[(i, z, l)] + \
							theta_TC_C[(i, z, l)] + \
							theta_TF_F[(i, z, l)] + \
							bias[(0, z, l)]) <= Mpre * (1 - v_T[i]), f"MILP-2LM-M-GC-(68)-{z}-{l}-{i}-1-1_ASGM{asgml.ind}"
					MILP += theta_T_tmp[(i, z, l + 1)] - (pulp.lpSum(w[(0, l, z_p, z)] * theta_T[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							theta_T_minus[(i, z, l)] + \
							theta_T_plus[(i, z, l)] + \
							theta_CT_C[(i, z, l)] + \
							theta_TC_C[(i, z, l)] + \
							theta_TF_F[(i, z, l)] + \
							bias[(0, z, l)]) >= -Mpre * (1 - v_T[i]), f"MILP-2LM-M-GC-(68)-{z}-{l}-{i}-1-2_ASGM{asgml.ind}"

					MILP += theta_T[(i, z, l + 1)] - kappa * theta_T_tmp[(i, z, l + 1)] >= -Mpre * delta_T_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-T-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_T[(i, z, l + 1)] - kappa * theta_T_tmp[(i, z, l + 1)] <= Mpre * delta_T_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-T-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_T[(i, z, l + 1)] - theta_T_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_T_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-T-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_T[(i, z, l + 1)] - theta_T_tmp[(i, z, l + 1)] <= Mpre * (1 - delta_T_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-T-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_T_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_T_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-T-3-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_T_tmp[(i, z, l + 1)] <= Mpre * delta_T_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-T-3-{z}-{l}-{i}-2_ASGM{asgml.ind}"

					MILP += theta_T[(i, z, l + 1)] <= Mth * v_T[i] , f"MILP-2LM-M-GC-(68)-{z}-{l}-{i}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_T[(i, z, l + 1)] >= -Mth * v_T[i], f"MILP-2LM-M-GC-(68)-{z}-{l}-{i}-3_ASGM{asgml.ind}" # by Zhu
				if (0, z, l) in theta_T:
					MILP += theta_T[(0, z, l)] == 0, f"MILP-2LM-M-GC-(68)-{z}-{l}-3_ASGM{asgml.ind}"
				if (t_T + 1, z, l) in theta_T:
					MILP += theta_T[(t_T + 1, z, l)] == 0, f"MILP-2LM-M-GC-(68)-{z}-{l}-4_ASGM{asgml.ind}"

		# -------- Constraint (73) --------
		for z in range(1, h_size + 1):
			for l in range(0, (l_max - 1) + 1):
				for i in range(1, t_F + 1):
					Mpre = m_pre("F", i, l + 1, z)
					Mth = m_theta("F", i, l + 1, z)
					MILP += theta_F_tmp[(i, z, l + 1)] - (pulp.lpSum(w[(0, l, z_p, z)] * theta_F[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							theta_F_minus[(i, z, l)] + \
							theta_F_plus[(i, z, l)] + \
							theta_CF_C[(i, z, l)] + \
							theta_TF_T[(i, z, l)] + \
							bias[(0, z, l)]) <= Mpre * (1 - v_F[i]), f"MILP-2LM-M-GC-(69)-{z}-{l}-{i}-1-1_ASGM{asgml.ind}"
					MILP += theta_F_tmp[(i, z, l + 1)] - (pulp.lpSum(w[(0, l, z_p, z)] * theta_F[(i, z_p, l)]
								for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
							theta_F_minus[(i, z, l)] + \
							theta_F_plus[(i, z, l)] + \
							theta_CF_C[(i, z, l)] + \
							theta_TF_T[(i, z, l)] + \
							bias[(0, z, l)]) >= -Mpre * (1 - v_F[i]), f"MILP-2LM-M-GC-(69)-{z}-{l}-{i}-1-2_ASGM{asgml.ind}"

					MILP += theta_F[(i, z, l + 1)] - kappa * theta_F_tmp[(i, z, l + 1)] >= -Mpre * delta_F_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-F-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_F[(i, z, l + 1)] - kappa * theta_F_tmp[(i, z, l + 1)] <= Mpre * delta_F_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-F-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_F[(i, z, l + 1)] - theta_F_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_F_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-F-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_F[(i, z, l + 1)] - theta_F_tmp[(i, z, l + 1)] <= Mpre * (1 - delta_F_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-F-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_F_tmp[(i, z, l + 1)] >= -Mpre * (1 - delta_F_tmp[(i, z, l + 1)]), f"MILP-GC-LeakyReLU-F-3-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_F_tmp[(i, z, l + 1)] <= Mpre * delta_F_tmp[(i, z, l + 1)], f"MILP-GC-LeakyReLU-C-F-{z}-{l}-{i}-2_ASGM{asgml.ind}"

					MILP += theta_F[(i, z, l + 1)] <= Mth * v_F[i], f"MILP-2LM-M-GC-(69)-{z}-{l}-{i}-2_ASGM{asgml.ind}"
					MILP += theta_F[(i, z, l + 1)] >= -Mth * v_F[i], f"MILP-2LM-M-GC-(69)-{z}-{l}-{i}-3_ASGM{asgml.ind}" # by Zhu
				if (0, z, l) in theta_F:
					MILP += theta_F[(0, z, l)] == 0, f"MILP-2LM-M-GC-(69)-{z}-{l}-3_ASGM{asgml.ind}"
				if (t_F + 1, z, l) in theta_F:
					MILP += theta_F[(t_F + 1, z, l)] == 0, f"MILP-2LM-M-GC-(69)-{z}-{l}-4_ASGM{asgml.ind}"

		# -------- Constraint (74) --------
		for z in range(1, h_size + 1):
			for l in range(0, l_max):
				for k in range(k_C_tilde + 1, m_C + 1): # range modified
					Mminus = m_msg("C", tail_C[k], l, z)
					Mplus = m_msg("C", head_C[k], l, z)
					MILP += theta_C_minus[(k, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(tail_C[k], z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mminus * (1 - e_C[k]), f"MILP-2LM-M-GC-(70)-1-{z}-{l}-{k}-1_ASGM{asgml.ind}"
					MILP += theta_C_minus[(k, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(tail_C[k], z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mminus * (1 - e_C[k]), f"MILP-2LM-M-GC-(70)-1-{z}-{l}-{k}-2_ASGM{asgml.ind}"
					MILP += theta_C_plus[(k, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(head_C[k], z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mplus * (1 - e_C[k]), f"MILP-2LM-M-GC-(70)-2-{z}-{l}-{k}-1_ASGM{asgml.ind}"
					MILP += theta_C_plus[(k, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(head_C[k], z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mplus * (1 - e_C[k]), f"MILP-2LM-M-GC-(70)-2-{z}-{l}-{k}-2_ASGM{asgml.ind}"
		for k in range(k_C_tilde + 1, m_C + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Mminus = m_msg("C", tail_C[k], l, z)
					Mplus = m_msg("C", head_C[k], l, z)
					MILP += theta_C_minus[(k, z, l)] <= Mminus * e_C[k], f"MILP-2LM-M-GC-(70)-3-{k}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_C_plus[(k, z, l)] <= Mplus * e_C[k], f"MILP-2LM-M-GC-(70)-3-{k}-{z}-{l}-2_ASGM{asgml.ind}"
					MILP += theta_C_minus[(k, z, l)] >= -Mminus * e_C[k], f"MILP-2LM-M-GC-(70)-4-{k}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_C_plus[(k, z, l)] >= -Mplus * e_C[k], f"MILP-2LM-M-GC-(70)-4-{k}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (75) --------
		for z in range(1, h_size + 1):
			for l in range(0, l_max):
				for i in range(2, t_T + 1):
					Mminus = m_msg("T", i - 1, l, z)
					MILP += theta_T_minus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i - 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mminus * (1 - e_T[i]), f"MILP-2LM-M-GC-(71)-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_T_minus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i - 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mminus * (1 - e_T[i]), f"MILP-2LM-M-GC-(71)-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
				for i in range(1, t_T):
					Mplus = m_msg("T", i + 1, l, z)
					MILP += theta_T_plus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i + 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mplus * (1 - e_T[i + 1]), f"MILP-2LM-M-GC-(71)-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_T_plus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i + 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mplus * (1 - e_T[i + 1]), f"MILP-2LM-M-GC-(71)-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
		for i in range(1, t_T + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Mminus = m_msg("T", i - 1, l, z) if i >= 2 else M_msg_layer[l]
					Mplus = m_msg("T", i + 1, l, z) if i <= t_T - 1 else M_msg_layer[l]
					MILP += theta_T_minus[(i, z, l)] <= Mminus * e_T[i], f"MILP-2LM-M-GC-(71)-3-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_T_plus[(i, z, l)] <= Mplus * e_T[i + 1], f"MILP-2LM-M-GC-(71)-3-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_T_minus[(i, z, l)] >= -Mminus * e_T[i], f"MILP-2LM-M-GC-(71)-4-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_T_plus[(i, z, l)] >= -Mplus * e_T[i + 1], f"MILP-2LM-M-GC-(71)-4-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (76) --------
		for z in range(1, h_size + 1):
			for l in range(0, l_max):
				for i in range(2, t_F + 1):
					Mminus = m_msg("F", i - 1, l, z)
					MILP += theta_F_minus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i - 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mminus * (1 - e_F[i]), f"MILP-2LM-M-GC-(72)-1-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_F_minus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i - 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mminus * (1 - e_F[i]), f"MILP-2LM-M-GC-(72)-1-{z}-{l}-{i}-2_ASGM{asgml.ind}"
				for i in range(1, t_F):
					Mplus = m_msg("F", i + 1, l, z)
					MILP += theta_F_plus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i + 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) <= \
								Mplus * (1 - e_F[i + 1]), f"MILP-2LM-M-GC-(72)-2-{z}-{l}-{i}-1_ASGM{asgml.ind}"
					MILP += theta_F_plus[(i, z, l)] - pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i + 1, z_p, l)]
									for z_p in range(1, (z_max if l == 0 else h_size) + 1)) >= \
								-Mplus * (1 - e_F[i + 1]), f"MILP-2LM-M-GC-(72)-2-{z}-{l}-{i}-2_ASGM{asgml.ind}"
		for i in range(1, t_F + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Mminus = m_msg("F", i - 1, l, z) if i >= 2 else M_msg_layer[l]
					Mplus = m_msg("F", i + 1, l, z) if i <= t_F - 1 else M_msg_layer[l]
					MILP += theta_F_minus[(i, z, l)] <= Mminus * e_F[i], f"MILP-2LM-M-GC-(72)-3-{i}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_F_plus[(i, z, l)] <= Mplus * e_F[i + 1], f"MILP-2LM-M-GC-(72)-3-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_F_minus[(i, z, l)] >= -Mminus * e_F[i], f"MILP-2LM-M-GC-(72)-4-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_F_plus[(i, z, l)] >= -Mplus * e_F[i + 1], f"MILP-2LM-M-GC-(72)-4-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (77) --------
		for k in range(1, k_C + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for i in range(1, t_T + 1):
						Mt = m_msg("T", i, l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mt * (1 - chi_T[(i, k)] + e_T[i]) <= \
									theta_CT_T[(k, z, l)], f"MILP-2LM-M-GC-(73)-1-{k}-{z}-{l}-{i}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mt * (1 - chi_T[(i, k)] + e_T[i]) >= \
									theta_CT_T[(k, z, l)], f"MILP-2LM-M-GC-(73)-1-{k}-{z}-{l}-{i}-2_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mt * (1 - chi_T[(i, k)] + e_T[i + 1]) <= \
									theta_TC_T[(k, z, l)], f"MILP-2LM-M-GC-(73)-2-{k}-{z}-{l}-{i}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mt * (1 - chi_T[(i, k)] + e_T[i + 1]) >= \
									theta_TC_T[(k, z, l)], f"MILP-2LM-M-GC-(73)-2-{k}-{z}-{l}-{i}-2_ASGM{asgml.ind}"
		for k in range(1, k_C + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_CT_T[(k, z, l)] <= Msel * delta_chi_T[k], f"MILP-2LM-M-GC-(73)-3-{k}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_TC_T[(k, z, l)] <= Msel * delta_chi_T[k], f"MILP-2LM-M-GC-(73)-3-{k}-{z}-{l}-2_ASGM{asgml.ind}"
					MILP += theta_CT_T[(k, z, l)] >= -Msel * delta_chi_T[k], f"MILP-2LM-M-GC-(73)-4-{k}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TC_T[(k, z, l)] >= -Msel * delta_chi_T[k], f"MILP-2LM-M-GC-(73)-4-{k}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (78) --------
		for i in range(1, t_T + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for k in range(1, k_C + 1):
						Mc_tail = m_msg("C", tail_C[k], l, z)
						Mc_head = m_msg("C", head_C[k], l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(tail_C[k], z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mc_tail * (1 - chi_T[(i, k)] + e_T[i]) <= \
									theta_CT_C[(i, z, l)], f"MILP-2LM-M-GC-(74)-1-{i}-{z}-{l}-{k}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(tail_C[k], z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mc_tail * (1 - chi_T[(i, k)] + e_T[i]) >= \
									theta_CT_C[(i, z, l)], f"MILP-2LM-M-GC-(74)-1-{i}-{z}-{l}-{k}-2_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(head_C[k], z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mc_head * (1 - chi_T[(i, k)] + e_T[i + 1]) <= \
									theta_TC_C[(i, z, l)], f"MILP-2LM-M-GC-(74)-2-{i}-{z}-{l}-{k}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(head_C[k], z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mc_head * (1 - chi_T[(i, k)] + e_T[i + 1]) >= \
									theta_TC_C[(i, z, l)], f"MILP-2LM-M-GC-(74)-2-{i}-{z}-{l}-{k}-2_ASGM{asgml.ind}"
		for i in range(1, t_T + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_CT_C[(i, z, l)] <= Msel * (1 - e_T[i]), f"MILP-2LM-M-GC-(74)-3-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TC_C[(i, z, l)] <= Msel * (1 - e_T[i + 1]), f"MILP-2LM-M-GC-(74)-3-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_CT_C[(i, z, l)] >= -Msel * (1 - e_T[i]), f"MILP-2LM-M-GC-(74)-5-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TC_C[(i, z, l)] >= -Msel * (1 - e_T[i + 1]), f"MILP-2LM-M-GC-(74)-5-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

					MILP += theta_CT_C[(i, z, l)] <= Msel * v_T[i], f"MILP-2LM-M-GC-(74)-4-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TC_C[(i, z, l)] <= Msel * v_T[i], f"MILP-2LM-M-GC-(74)-4-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_CT_C[(i, z, l)] >= -Msel * v_T[i], f"MILP-2LM-M-GC-(74)-6-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TC_C[(i, z, l)] >= -Msel * v_T[i], f"MILP-2LM-M-GC-(74)-6-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (79) --------
		for c in range(1, t_C_tilde + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for i in range(1, t_F + 1):
						Mf = m_msg("F", i, l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mf * (1 - chi_F[(i, c)] + e_F[i]) <= \
									theta_CF_F[(c, z, l)], f"MILP-2LM-M-GC-(75)-1-{c}-{z}-{l}-{i}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(i, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mf * (1 - chi_F[(i, c)] + e_F[i]) >= \
									theta_CF_F[(c, z, l)], f"MILP-2LM-M-GC-(75)-1-{c}-{z}-{l}-{i}-2_ASGM{asgml.ind}"
		for c in range(1, t_C_tilde + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_CF_F[(c, z, l)] <= Msel * delta_chi_F[c], f"MILP-2LM-M-GC-(75)-2-{c}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_CF_F[(c, z, l)] >= -Msel * delta_chi_F[c], f"MILP-2LM-M-GC-(75)-2-{c}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (80) --------
		for i in range(1, t_F + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for c in range(1, t_C_tilde + 1):
						Mc = m_msg("C", c, l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(c, z_p, l)] # should be c
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mc * (1 - chi_F[(i, c)] + e_F[i]) <= \
									theta_CF_C[(i, z, l)], f"MILP-2LM-M-GC-(76)-1-{i}-{z}-{l}-{c}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_C[(c, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mc * (1 - chi_F[(i, c)] + e_F[i]) >= \
									theta_CF_C[(i, z, l)], f"MILP-2LM-M-GC-(76)-1-{i}-{z}-{l}-{c}-2_ASGM{asgml.ind}"
		for i in range(1, t_F + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_CF_C[(i, z, l)] <= Msel * pulp.lpSum(chi_F[(i, c)]
								for c in range(1, t_C_tilde + 1)), f"MILP-2LM-M-GC-(76)-2-{i}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_CF_C[(i, z, l)] >= -Msel * pulp.lpSum(chi_F[(i, c)]
								for c in range(1, t_C_tilde + 1)), f"MILP-2LM-M-GC-(76)-2-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_CF_C[(i, z, l)] <= Msel * (1 - e_F[i]), f"MILP-2LM-M-GC-(76)-3-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_CF_C[(i, z, l)] >= -Msel * (1 - e_F[i]), f"MILP-2LM-M-GC-(76)-3-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (81) --------
		for i in range(1, t_T + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for j in range(1, t_F + 1):
						Mf = m_msg("F", j, l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(j, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mf * (1 - chi_F[(j, t_C_tilde + i)] + e_F[j]) <= \
									theta_TF_F[(i, z, l)], f"MILP-2LM-M-GC-(77)-1-{i}-{z}-{l}-{j}-1_ASGM{asgml.ind}"
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_F[(j, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mf * (1 - chi_F[(j, t_C_tilde + i)] + e_F[j]) >= \
									theta_TF_F[(i, z, l)], f"MILP-2LM-M-GC-(77)-1-{i}-{z}-{l}-{j}-2_ASGM{asgml.ind}"
		for i in range(1, t_T + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_TF_F[(i, z, l)] <= Msel * delta_chi_F[t_C_tilde + i], f"MILP-2LM-M-GC-(77)-2-{i}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_TF_F[(i, z, l)] >= -Msel * delta_chi_F[t_C_tilde + i], f"MILP-2LM-M-GC-(77)-2-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (82) --------
		for i in range(1, t_F + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					for j in range(1, t_T + 1):
						Mt = m_msg("T", j, l, z)
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(j, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) - \
								Mt * (1 - chi_F[(i, t_C_tilde + j)] + e_F[i]) <= \
									theta_TF_T[(i, z, l)], f"MILP-2LM-M-GC-(78)-1-{i}-{z}-{l}-{j}-1_ASGM{asgml.ind}" # by Zhu
						MILP += pulp.lpSum(w[(1, l, z_p, z)] * theta_T[(j, z_p, l)]
										for z_p in range(1, (z_max if l == 0 else h_size) + 1)) + \
								Mt * (1 - chi_F[(i, t_C_tilde + j)] + e_F[i]) >= \
									theta_TF_T[(i, z, l)], f"MILP-2LM-M-GC-(78)-1-{i}-{z}-{l}-{j}-2_ASGM{asgml.ind}" # by Zhu
		for i in range(1, t_F + 1):
			for z in range(1, h_size + 1):
				for l in range(0, l_max):
					Msel = M_msg_layer[l]
					MILP += theta_TF_T[(i, z, l)] <= Msel * pulp.lpSum(chi_F[(i, t_C_tilde + j)]
								for j in range(1, t_T + 1)), f"MILP-2LM-M-GC-(78)-2-{i}-{z}-{l}-1_ASGM{asgml.ind}"
					MILP += theta_TF_T[(i, z, l)] >= -Msel * pulp.lpSum(chi_F[(i, t_C_tilde + j)]
								for j in range(1, t_T + 1)), f"MILP-2LM-M-GC-(78)-2-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TF_T[(i, z, l)] <= Msel * (1 - e_F[i]), f"MILP-2LM-M-GC-(78)-3-{i}-{z}-{l}-1_ASGM{asgml.ind}" # by Zhu
					MILP += theta_TF_T[(i, z, l)] >= -Msel * (1 - e_F[i]), f"MILP-2LM-M-GC-(78)-3-{i}-{z}-{l}-2_ASGM{asgml.ind}" # by Zhu

		# -------- Constraint (83) --------
		for p in range(1, p_cv + 1):
			Mread = M_cv[p]
			MILP += pulp.lpSum(w_cv[(z, p)] * theta_C[(i, z, l_max)]
						for i in range(1, t_C + 1) for z in range(1, h_size + 1)) + \
					pulp.lpSum(w_cv[(z, p)] * theta_T[(i, z, l_max)]
						for i in range(1, t_T + 1) for z in range(1, h_size + 1)) + \
					pulp.lpSum(w_cv[(z, p)] * theta_F[(i, z, l_max)]
						for i in range(1, t_F + 1) for z in range(1, h_size + 1)) == \
					theta_cv_tmp[p], f"MILP-2LM-M-GC-(79)-{p}"

			MILP += theta_cv[p] - kappa * theta_cv_tmp[p] >= -Mread * delta_cv_tmp[p], f"MILP-GC-LeakyReLU-p-1-{p}-1_ASGM{asgml.ind}"
			MILP += theta_cv[p] - kappa * theta_cv_tmp[p] <= Mread * delta_cv_tmp[p], f"MILP-GC-LeakyReLU-p-1-{p}-2_ASGM{asgml.ind}"
			MILP += theta_cv[p] - theta_cv_tmp[p] >= -Mread * (1 - delta_cv_tmp[p]), f"MILP-GC-LeakyReLU-p-2-{p}-1_ASGM{asgml.ind}"
			MILP += theta_cv[p] - theta_cv_tmp[p] <= Mread * (1 - delta_cv_tmp[p]), f"MILP-GC-LeakyReLU-p-2-{p}-2_ASGM{asgml.ind}"
			MILP += theta_cv_tmp[p] >= -Mread * (1 - delta_cv_tmp[p]), f"MILP-GC-LeakyReLU-p-3-{p}-1_ASGM{asgml.ind}"
			MILP += theta_cv_tmp[p] <= Mread * delta_cv_tmp[p], f"MILP-GC-LeakyReLU-p-3-{p}-2_ASGM{asgml.ind}"

		return MILP

	def set_up_x_hat_variables(self):
		pass

	def add_constraints_C1(self, MILP, asgml, sg):
		MILP = self.add_constraints_graph_convolution(
				MILP, asgml,
				sg.t_C, sg.t_C_tilde, sg.t_T, sg.t_F, sg.k_C, sg.k_C_tilde, sg.m_C, sg.val, sg.Code_F, sg.Lambda_int,
				sg.I_a_minus, sg.I_a_plus, sg.I_b_minus, sg.I_b_plus, sg.F_C, sg.F_T, sg.F_F, sg.head_C, sg.tail_C,
				sg.delta_alpha_C, sg.delta_alpha_T, sg.delta_alpha_F, sg.delta_fr_C, sg.delta_fr_T, sg.delta_fr_F,
				sg.delta_beta_C, sg.delta_beta_T, sg.delta_beta_F, sg.delta_beta_plus, sg.delta_beta_minus, sg.delta_beta_in,
				sg.e_C, sg.e_T, sg.e_F, sg.delta_chi_T, sg.delta_chi_F, sg.chi_T, sg.chi_F, sg.v_T, sg.v_F,
				sg.alpha_C, sg.alpha_T, sg.alpha_F, sg.deg_C, sg.deg_T, sg.deg_F,
				sg.eledeg_C, sg.eledeg_T, sg.eledeg_F, sg.hyddeg_C, sg.hyddeg_T, sg.hyddeg_F,
				self.theta_cv[asgml.ind], self.theta_cv_tmp[asgml.ind], self.delta_cv_tmp[asgml.ind],
				self.theta_C[asgml.ind], self.theta_T[asgml.ind], self.theta_F[asgml.ind],
				self.theta_C_tmp[asgml.ind], self.theta_T_tmp[asgml.ind], self.theta_F_tmp[asgml.ind],
				self.delta_C_tmp[asgml.ind], self.delta_T_tmp[asgml.ind], self.delta_F_tmp[asgml.ind],
				self.theta_C_minus[asgml.ind], self.theta_C_plus[asgml.ind], self.theta_T_minus[asgml.ind],
				self.theta_T_plus[asgml.ind], self.theta_F_minus[asgml.ind], self.theta_F_plus[asgml.ind],
				self.theta_CT_T[asgml.ind], self.theta_TC_T[asgml.ind], self.theta_CT_C[asgml.ind], self.theta_TC_C[asgml.ind],
				self.theta_TF_F[asgml.ind], self.theta_CF_F[asgml.ind], self.theta_CF_C[asgml.ind], self.theta_TF_T[asgml.ind]
			)
		MILP = self.inv_ann.add_constraints_C1(MILP, asgml)

		for i in range(1, self.p_cv + 1):
			MILP += self.inv_ann.y[asgml.ind][(1, i)] == self.theta_cv[asgml.ind][i], f"GNN-ANN-connect-{i}_ASGM{asgml.ind}"

		return MILP

	def inspection(self, test_sdf):
		cg_list = chemicalgraph.construct_from_SDF_file(test_sdf, use_rdkit=False)
		cg_list.get_fringe_tree()
		cg_list.set_value("1", 0.0)
		cg_dataset = dataset.ChemicalGraph_Datasets(root='./data/')
		cg_dataset.construct_from_ChemicalGraph_list(cg_list, FT_use=True)
		ft_dataset = dataset.FringeTree_Datasets(root='./data')
		ft_dataset.construct_from_FringeTree_list(cg_list.fringe_tree_list)

		loader = DataLoader(cg_dataset, batch_size=len(cg_dataset))
		ft_dataloader = DataLoader(ft_dataset, batch_size=len(ft_dataset))
		for ft_dataset_batch in ft_dataloader:
			break

		self.GNN.eval()
		with torch.inference_mode():
			for data in loader:
				data = data.to(self.device)
				ft_dataset_batch.to(self.device)
				output = self.GNN(data, ft_dataset_batch)
				y_pred = output.cpu().numpy()

		return y_pred[0][0]
