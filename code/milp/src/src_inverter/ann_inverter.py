from .inverter_base import *
import pulp
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, ClassVar
import pandas as pd
import numpy as np
from src.utils import _read_clean_lines, _pop_int, _pop_ints, _pop_floats, _pop_float, run_with_retry

@dataclass(frozen=False, kw_only=True)
class ANNInverter(InverterBase):
	### An inverter class for ANN
	name: str
	_type: ClassVar[Literal["INV_ANN"]] = "INV_ANN"

	def __post_init__(self, name="ann"):
		self.output = dict()
		self.index_set = dict() # store the relation between the indices of fringe trees in fv files and in seed graph files, index_set[index_in_fv] = index_in_seed_graph
		self.x_hat = dict() # store the normalized feature vector value
		self.std_eps = 1e-5

		self.x = dict()
		self.y = dict()
		self.z = dict()

	@property
	def non_input_nodes(self):
		for layer in self.layers[1:]:
			for node in layer:
				yield node

	def read_fringe_tree(self, fv_fringe_tree_filename):
		self.fv_fringe_tree = dict()
		with open(fv_fringe_tree_filename, 'r') as f:
			lines = f.readlines()
			if lines[0][0] == 'F':
				lines = lines[1:]
			for line in lines:
				if len(line.split(",")) < 4:
					continue
				ind = int(line.split(",")[0])
				str1 = line.split(",")[1]
				str2 = line.split(",")[2]
				str3 = line.split(",")[3].replace('\n', '')
				self.fv_fringe_tree[ind] = (str1, str2, str3)

	def prepare_fringe_tree_index_set(self, asgml_ind, strF, init=True):
		if asgml_ind not in self.index_set or init:
			self.index_set[asgml_ind] = dict()
		for (key, f) in strF.items():
			for (ind, fv_fringe_tree) in self.fv_fringe_tree.items():
				if fv_fringe_tree == f:
					self.index_set[asgml_ind][ind] = key
					break

	def _initialize(self, weight_tensor, bias_matrix):
		self.nodes = list()
		self.layers = list()
		self.weights = dict()
		self.biases = dict()
		self.in_neighbors = dict()

		for l, bias_layer in enumerate(bias_matrix):
			layer_nodes = list()
			for k, bias in enumerate(bias_layer):
				node = (l + 1, k + 1)
				self.nodes.append(node)
				self.biases[node] = bias
				layer_nodes.append(node)
			self.layers.append(layer_nodes)
		for l, layer in enumerate(self.layers):
			try:
				for node in self.layers[l+1]:
					self.in_neighbors[node] = layer
			except IndexError:
				pass
			except ValueError:
				raise ValueError("Received Value Error in distributing ANN nodes by layers")

		for l, weight_matrix in enumerate(weight_tensor):
			for i, weight_row in enumerate(weight_matrix):
				for j, weight in enumerate(weight_row):
					u = (l + 1, i + 1)
					v = (l + 2, j + 1)
					self.weights[(u, v)] = weight

	def _linear_interval(self, terms, bias=0.0):
		"""
		Compute interval bounds for sum_i w_i y_i + bias.

		terms: list of (weight, lower, upper)
		"""
		L = float(bias)
		U = float(bias)

		for w, yL, yU in terms:
			w = float(w)
			yL = float(yL)
			yU = float(yU)

			if w >= 0:
				L += w * yL
				U += w * yU
			else:
				L += w * yU
				U += w * yL

		return L, U


	def _initialize_constants(self, input_bounds=None, safety=1.0):
		"""
		Compute nodewise interval bounds for ANN.

		x_bounds[v] = bounds for preactivation x_v = W y + b.
		y_bounds[v] = bounds for postactivation y_v.

		For hidden layers:
			y_v = ReLU(x_v)

		For output layer:
			y_v = x_v
		"""
		x_bounds = dict()
		y_bounds = dict()
		relu_state = dict()

		# ------------------------------------------------------------
		# Input layer bounds
		# ------------------------------------------------------------
		# If input_bounds is not provided, use the old safe fallback [-5, 5].
		# If your normalized descriptors are guaranteed in [0, 1],
		# pass input_bounds={i: (0.0, 1.0)} or set the default below to [0,1].
		for i, node in enumerate(self.layers[0], start=1):
			if input_bounds is None:
				L, U = -5.0, 5.0
			else:
				# Accept both i and node as keys.
				L, U = input_bounds.get(i, input_bounds.get(node, (-5.0, 5.0)))

			L = float(L)
			U = float(U)
			if L > U:
				raise ValueError(f"Invalid input bound for ANN input node {node}: ({L}, {U})")

			y_bounds[node] = (L, U)

		# ------------------------------------------------------------
		# Hidden/output layer bounds
		# ------------------------------------------------------------
		for layer in self.layers[1:]:
			for v in layer:
				terms = [
					(self.weights[(u, v)], y_bounds[u][0], y_bounds[u][1])
					for u in self.in_neighbors[v]
				]
				Lx, Ux = self._linear_interval(terms, bias=self.biases[v])

				# Safety padding, only outward.
				if safety != 1.0:
					Lx = Lx * safety if Lx < 0 else Lx / safety
					Ux = Ux * safety if Ux > 0 else Ux / safety

				x_bounds[v] = (Lx, Ux)

				if v in self.layers[-1]:
					# Output layer is linear.
					y_bounds[v] = (Lx, Ux)
					relu_state[v] = "linear_output"
				else:
					if Ux <= 0:
						# ReLU always inactive.
						y_bounds[v] = (0.0, 0.0)
						relu_state[v] = "inactive"
					elif Lx >= 0:
						# ReLU always active.
						y_bounds[v] = (Lx, Ux)
						relu_state[v] = "active"
					else:
						# Ambiguous ReLU.
						y_bounds[v] = (0.0, Ux)
						relu_state[v] = "ambiguous"

		# Keep old names as much as possible for compatibility.
		self.ann_x_bounds = x_bounds
		self.ann_y_bounds = y_bounds
		self.ann_relu_state = relu_state

		# # Optional compatibility dictionaries.
		# # ann_a[v] = (Lx, 0, Ux), ann_b[v] = (Ly, 0, Uy)
		# ann_a = {
		# 	v: (x_bounds[v][0], 0.0, x_bounds[v][1])
		# 	for v in self.non_input_nodes
		# }
		# ann_b = {
		# 	v: (y_bounds[v][0], 0.0, y_bounds[v][1])
		# 	for v in self.nodes
		# }

		# return ann_a, ann_b, None, None, None

	def set_up_inverter(self, invl):
		self.sdf_filename = invl.sdf_filename
		self.fv_gen_path = invl.fv_gen_path
		self._use_quadratic = invl._use_quadratic

		weight_filename = invl.weight_filename
		biases_filename = invl.biases_filename

		try:
			Q = _read_clean_lines(weight_filename)
			layer_sizes = _pop_ints(Q)
			self.layer_num = len(layer_sizes)
			self.num_K = layer_sizes[0]
			weight_tensor = list()
			for ell in layer_sizes[:-1]:
				weight_matrix = list()
				for _ in range(ell):
					weight_line = _pop_floats(Q)
					weight_matrix.append(weight_line)
				weight_tensor.append(weight_matrix)
			Q = _read_clean_lines(biases_filename)
			bias_matrix = list()
			# The input layer has no bias, and we define it to be 0
			bias_matrix.append([0] * layer_sizes[0])
			for ell in layer_sizes[1:]:
				bias_row = list()
				for _ in range(ell):
					bias_line = _pop_float(Q)
					bias_row.append(bias_line)
				bias_matrix.append(bias_row)
		except:
			raise ValueError(f"Error: Failed to read the inverter {self.name}!")

		self._initialize(weight_tensor, bias_matrix)
		input_bounds = {i: (-5.0, 5.0) for i in range(1, self.num_K + 1)} # safety bounds
		# self.ann_a, self.ann_b, _, _, _ = self._initialize_constants(input_bounds=input_bounds)
		self._initialize_constants(input_bounds=input_bounds)

		self.read_fringe_tree(invl.fringe_filename)
		self.prepare_fv_list(invl.desc_norm_selected_filename)
		self.prepare_desc_max_min(invl.desc_filename)
		self.prepare_value_max_min(invl.desc_norm_selected_filename, invl.values_filename)

	def set_up_inverter_for_GNN(self, input_bounds, weight_tensor, bias_matrix, layer_sizes):
		self.layer_num = len(layer_sizes)
		self._initialize(weight_tensor, bias_matrix)
		# self.ann_a, self.ann_b, _, _, _ = self._initialize_constants(input_bounds=input_bounds)
		self._initialize_constants(input_bounds=input_bounds)

	def set_up_variables(self, asgml):
		self.x[asgml.ind] = {node: pulp.LpVariable(f"ANN_x_{node}_ASGM{asgml.ind}",
					self.ann_x_bounds[node][0], self.ann_x_bounds[node][1],
					cat=pulp.LpContinuous)
			for node in self.non_input_nodes
		}

		self.y[asgml.ind] = {node: pulp.LpVariable(f"ANN_y_{node}_ASGM{asgml.ind}",
					self.ann_y_bounds[node][0], self.ann_y_bounds[node][1],
					cat=pulp.LpContinuous)
			for node in self.nodes
		}

		# Only ambiguous hidden ReLU nodes need binary variables.
		self.z[asgml.ind] = {node: pulp.LpVariable(f"ANN_z_{node}_ASGM{asgml.ind}", 0, 1, cat=pulp.LpBinary)
			for node in self.non_input_nodes if self.ann_relu_state[node] == "ambiguous"
		}

		### The output variable uses self.output as the variable name
		self.output[asgml.ind] = self.y[asgml.ind][(self.layer_num, 1)]

	def set_up_x_hat_variables(self, asgml):
		self.x_hat[asgml.ind] = {i: self.y[asgml.ind][(1, i)] for i in range(1, self.num_K + 1)}

	def add_constraints_C1(self, MILP, asgml):
		for v in self.non_input_nodes:
			in_u = self.in_neighbors[v]
			affine_expr = (pulp.lpSum(self.weights[(u, v)] * self.y[asgml.ind][u] for u in in_u) + self.biases[v])
			MILP += self.x[asgml.ind][v] == affine_expr, f"ANN_Affine_node_{v}_ASGM{asgml.ind}"
			L, U = self.ann_x_bounds[v]

			if v in self.layers[-1]:
				# Output layer is linear.
				MILP += self.y[asgml.ind][v] == self.x[asgml.ind][v], f"ANN_Output_linear_{v}_ASGM{asgml.ind}"
				continue

			state = self.ann_relu_state[v]

			if state == "inactive":
				# U <= 0, so ReLU(x) = 0.
				MILP += self.y[asgml.ind][v] == 0, f"ANN_ReLU_inactive_y_{v}_ASGM{asgml.ind}"
				# Optional, because x variable already has upper bound U <= 0.
				MILP += self.x[asgml.ind][v] <= 0, f"ANN_ReLU_inactive_x_{v}_ASGM{asgml.ind}"

			elif state == "active":
				# L >= 0, so ReLU(x) = x.
				MILP += self.y[asgml.ind][v] == self.x[asgml.ind][v], f"ANN_ReLU_active_{v}_ASGM{asgml.ind}"
				# Optional, because x variable already has lower bound L >= 0.
				MILP += self.x[asgml.ind][v] >= 0, f"ANN_ReLU_active_x_{v}_ASGM{asgml.ind}"

			else:
				# Ambiguous ReLU: L < 0 < U.
				z = self.z[asgml.ind][v]

				# y = max(0, x), ideal big-M formulation for known L < 0 < U.
				MILP += self.y[asgml.ind][v] >= self.x[asgml.ind][v], f"ANN_ReLU_amb_y_ge_x_{v}_ASGM{asgml.ind}"
				MILP += self.y[asgml.ind][v] >= 0, f"ANN_ReLU_amb_y_ge_0_{v}_ASGM{asgml.ind}"
				MILP += self.y[asgml.ind][v] <= self.x[asgml.ind][v] - L * (1 - z), f"ANN_ReLU_amb_y_ub_x_{v}_ASGM{asgml.ind}"
				MILP += self.y[asgml.ind][v] <= U * z, f"ANN_ReLU_amb_y_ub_0_{v}_ASGM{asgml.ind}"

		return MILP

	def _activition(self, x):
		return max(0, x)  # ReLU

	def _inspection(self, fv_test):
		cur_layer = self.layers[0]
		vec_y = np.array(fv_test)
		for next_layer in self.layers[1:]:
			weight_matrix = np.array([[self.weights[(u, v)] for u in cur_layer] for v in next_layer])
			bias_vector = np.array([self.biases[v] for v in next_layer])
			vec_x = np.dot(weight_matrix, vec_y) + bias_vector

			if next_layer == self.layers[-1]:
				vec_y = np.array([x for x in vec_x])
			else:
				vec_y = np.array([self._activition(x) for x in vec_x])

			cur_layer = next_layer
		insp_value = vec_y[0]
		return insp_value

	def inspection(self, asgml, fv_test, std_eps=None):
		if std_eps is None:
			std_eps = self.std_eps
		insp_value = self._inspection(fv_test)

		found_error = False
		for i in range(1, self.num_K + 1):
			if abs(self.x_hat[asgml.ind][i].value() - fv_test[i - 1]) > 2 * std_eps:
				print(f"[Assignment {asgml.ind}] [i = {i}, {self.fv_list[i - 1]}] Difference between fv_from_MILP ({self.x_hat[asgml.ind][i].value():.6f}) and fv_from_sdf ({fv_test[i - 1]:.6f}).")
				found_error = True
		if not found_error:
			print(f"[Assignment {asgml.ind}] No difference between feature vectors!")

		return insp_value
