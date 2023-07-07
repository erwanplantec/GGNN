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

class GNN(eqx.Module):
    
    """
    GNN update from https://arxiv.org/abs/1806.01261 (Algorithm 1)
        1. Update edges eij(t+1) = edge_fn(eij || hi || hj)
        2. aggregate edges to nodes aggr_edges = agg_fn(edges, graph.receivers)
        3. update nodes hi(t+1) = node_fn(hi, aggr_edges)
    """
    #-------------------------------------------------------------------
    node_fn: eqx.Module
    edge_fn: eqx.Module
    aggr_fn: t.Callable
    residual: bool
    #-------------------------------------------------------------------
    
    def __init__(self, node_fn: eqx.Module, edge_fn: eqx.Module, 
                 aggr_fn: t.Callable = jax.ops.segment_sum, residual: bool = True):
        
        self.node_fn = jax.vmap(node_fn)
        self.edge_fn = jax.vmap(edge_fn)
        self.aggr_fn = aggr_fn
        self.residual = residual
        
    #-------------------------------------------------------------------
        
    @ejit
    def __call__(self, graph: jraph.GraphsTuple, key: jr.PRNGKey):
        
        nodes, edges, receivers, senders, *_ = graph
        n_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        nodes_senders = nodes[senders]
        nodes_receivers = nodes[receivers]
        
        # 1. Update edges
        edges = self.edge_fn(
            cat([edges, nodes_receivers, nodes_senders])
        )
        edges = edges * graph.active_edges[:, None]
        
        # 2. Update nodes
        aggr_edges = self.aggr_fn(edges, receivers, n_nodes)
        dnodes = self.node_fn(
            cat([aggr_edges, nodes])
        )
        dnodes = dnodes * graph.active_nodes[:, None]
        if self.residual:
            nodes = nodes + dnodes
        else :
            nodes = dnodes
        
        return graph._replace(nodes=nodes, 
                              edges=edges)
