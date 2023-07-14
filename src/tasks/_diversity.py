from src.models._graph import GGraph
from src.models._utils import rollout

import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import typing as t
from functools import partial

def dist(x, y):
	return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=-1))

def kl(p, q):
	return jnp.sum(p * jnp.log(p/(q+1e-8)))

def knn_sparsity(x, n, k, dist_fn=dist):
	dists = dist_fn(x, x)
	res = 0.
	for i in range(n):
		idxs = jnp.argsort(dists[i])
		knn = idxs[1:k+1]
		res += jnp.mean(dists[i, knn])
	return res / n

def C(X, Y):
	"""measure compositionality of process f such that Y = f(X)
	X and Y are 2D arrays"""
	dists_X = jnp.sqrt(jnp.sum(X[:, None, :] - X[None, :, :], axis=-1)**2)
	dists_Y = jnp.sqrt(jnp.sum(Y[:, None, :] - Y[None, :, :], axis=-1)**2)

	dists_X = jnp.reshape(dists_X, -1)
	dists_Y = jnp.reshape(dists_Y, -1)

	return jnp.corrcoef(dists_X, dists_Y)[0, -1]


def DiversityTask(statics: t.Collection, init_graph: GGraph, metrics_fn: t.Callable,
				  sample_size: int, grow_steps: int=50, k: int=5, dist_fn: t.Callable=dist,
				  compositionality: bool=False):

	n_nodes = init_graph.active_nodes.sum().astype(int)
	max_nodes = init_graph.nodes.shape[0]
	node_features = init_graph.nodes.shape[-1]
	n_edges = init_graph.active_edges.sum().astype(int)
	max_edges = init_graph.edges.shape[0]
	edge_features = init_graph.edges.shape[-1]

	map_axes = GGraph(nodes=0, edges=None, n_node=None, n_edge=None, 
					  receivers=None, senders=None, active_nodes=None, 
					  active_edges=None, globals=None, time=None)

	#-------------------------------------------------------------------
	def sample_graphs(key):
		return init_graph._replace(nodes=jr.normal(key, (sample_size, *init_graph.nodes.shape)))
	#-------------------------------------------------------------------

	@partial(jax.vmap, in_axes=(None, 0))
	def _eval(key: jr.PRNGKey, params: t.Collection):
		key, key_sample, key_rollout = jr.split(key, 3)
		model = eqx.combine(params, statics)
		init_graphs = sample_graphs(key_sample)
		end_graphs = eqx.filter_vmap(model, in_axes=(map_axes, None, None, None, None))(init_graphs, 
											key_rollout,
											grow_steps,
											False, False)
		metrics = metrics_fn(end_graphs) 
		sparsity = knn_sparsity(metrics, n=sample_size, k=k, dist_fn=dist_fn)
		fitness = sparsity
		if compositionality:
			Z = init_graphs.nodes[:, :n_nodes].reshape((sample_size, -1))
			compo = C(metrics, Z) * (1.-jnp.allclose(jnp.var(metrics, axis=0), 0.).astype(float))
			fitness = fitness + compo
		return - fitness
	#-------------------------------------------------------------------

	return _eval
