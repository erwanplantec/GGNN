from src.metrics import degrees, in_degrees, out_degrees
from src.models._graph import GGraph

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import typing as t

default_node_metrics = [out_degrees, in_degrees]

class NodeMetricsInjection(eqx.Module):

	#-------------------------------------------------------------------
	metric_fn: t.Callable
	#-------------------------------------------------------------------

	def __init__(self, metrics: t.Iterable[t.Callable]=default_node_metrics, index: int=0):
		n = len(metrics)
		@jax.jit
		def apply(graph: GGraph):
			res = jnp.dstack([metric(graph) for metric in metrics])[0]
			res = res * graph.active_nodes[..., None]
			return graph._replace(nodes=graph.nodes.at[:, index:index+n].set(res))
		self.metric_fn = apply

	#-------------------------------------------------------------------

	def __call__(self, graph: GGraph, key: jr.PRNGKey):
		
		return self.metric_fn(graph)



default_edge_metrics = []

class EdgeMetricsInjection(eqx.Module):

	#-------------------------------------------------------------------
	metric_fn: t.Callable
	#-------------------------------------------------------------------

	def __init__(self, metrics: t.Iterable[t.Callable] = default_edge_metrics):
		n = len(metrics)
		@jax.jit
		def apply(graph: GGraph):
			res = jnp.dstack([metric(graph) for metric in metrics])[0]
			res = res * graph.active_edges[..., None]
			return graph._replace(edges=graph.edges.at[:, -n:].set(res))
		self.metric_fn = apply

	#-------------------------------------------------------------------

	def __call__(self, graph: GGraph, key: jr.PRNGKey):
		
		return self.metric_fn(graph)