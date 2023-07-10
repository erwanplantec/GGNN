import jax
import jax.numpy as jnp
import jax.random as jr

import jraph
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap

from functools import partial
import typing as t

from src.models._utils import *
from src.models._graph import GGraph

class GNCA(eqx.Module):
    
    """
    https://arxiv.org/abs/2110.14237
    """
    #-------------------------------------------------------------------
    node_fn: eqx.Module
    message_fn: eqx.Module
    aggr_fn: t.Callable
    residual: bool
    use_edges: bool
    #-------------------------------------------------------------------
    
    def __init__(self, node_fn: t.Callable, message_fn: t.Callable, 
                 aggr_fn: t.Callable=jax.ops.segment_sum, residual: bool=True,
                 use_edges: bool=False):
        
        self.node_fn = node_fn
        self.message_fn = message_fn
        self.aggr_fn = aggr_fn
        self.residual = residual
        self.use_edges = use_edges
        
    #-------------------------------------------------------------------
        
    @ejit
    def __call__(self, graph: jraph.GraphsTuple, key: jr.PRNGKey):
        
        if self.use_edges:
            #1. Compute messages
            m = jax.vmap(self.message_fn)(graph.nodes)
            #2. Aggregate messages
            m = self.aggr_fn(m[graph.senders], graph.receivers, graph.nodes.shape[0])
        else:
            #1. Compute messages
            m = jax.vmap(self.message_fn)(jnp.concatenate([graph.nodes[graph.senders], edges], axis=-1))
            #2. Aggregate messages
            m = self.aggr_fn(m, graph.receivers, graph.nodes.shape[0])

        #3. Update nodes
        nodes = jax.vmap(self.node_fn)(jnp.concatenate([graph.nodes, m], axis=-1))

        if self.residual:
            return graph._replace(nodes=graph.nodes+nodes)
        else: 
            return graph._replace(nodes=nodes)
