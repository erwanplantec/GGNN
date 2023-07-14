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
    edge_fn: t.Callable
    #-------------------------------------------------------------------
    
    def __init__(self, node_fn: t.Callable, message_fn: t.Callable, 
                 aggr_fn: t.Callable=jax.ops.segment_sum, residual: bool=True,
                 use_edges: bool=False, edge_fn: t.Callable=None):
        
        self.node_fn = node_fn
        self.message_fn = message_fn
        self.aggr_fn = aggr_fn
        self.residual = residual
        self.use_edges = use_edges
        if use_edges:
            assert edge_fn is not None, "edge_fn must be provided is use_edges is True"
        self.edge_fn = edge_fn
        
    #-------------------------------------------------------------------
        
    @ejit
    def __call__(self, graph: jraph.GraphsTuple, key: jr.PRNGKey):
        
        if not self.use_edges:
            edges = graph.edges
            #1. Compute messages
            m = jax.vmap(self.message_fn)(graph.nodes)
            #2. Aggregate messages
            m = self.aggr_fn(m[graph.senders], graph.receivers, graph.nodes.shape[0])
        else:
            #1. Update edges
            edges = self.edge_fn(
                jnp.concatenate(
                    [graph.edges, graph.nodes[graph.receivers], graph.nodes[graph.senders]], 
                    axis=-1
                )
            )
            #2. Compute messages
            m = jax.vmap(self.message_fn)(jnp.concatenate([graph.nodes[graph.senders], graph.edges], axis=-1))
            #3. Aggregate messages
            m = self.aggr_fn(m, graph.receivers, graph.nodes.shape[0])

        #3. Update nodes
        nodes = jax.vmap(self.node_fn)(jnp.concatenate([graph.nodes, m], axis=-1))

        if self.residual:
            return graph._replace(nodes=graph.nodes+nodes, edges=graph.edges+edges)
        else: 
            return graph._replace(nodes=nodes, edges=edges)
