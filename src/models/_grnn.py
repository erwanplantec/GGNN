from src.models._graph import GGraph

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import equinox.nn as nn



class GRNN(eqx.Module):

	"""
	"""
	#-------------------------------------------------------------------
	rnn: eqx.Module
	aggr_fn: t.Callable
	#-------------------------------------------------------------------

	def __init__(self, rnn: eqx.Module, aggregation_fn: t.Callable=jax.ops.segment_sum):
		
		self.rnn = rnn
		self.aggr_fn = aggregation_fn

	#-------------------------------------------------------------------

	def __call__(self, graph: GGraph, key: jr.PRNGKey):
		
		nodes, edges, *_ = graph

		aggr_edges = self.aggr_fn(edges, graph.receivers, nodes.shape[0])
		nodes = self.rnn(aggr_edges, nodes)

		return graph._replace(nodes=nodes)
