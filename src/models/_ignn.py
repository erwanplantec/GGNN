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

class IGNN(eqx.Module):

    """
    Interaction based GNN (https://arxiv.org/abs/1612.00222)
        1. Update edges attributes (relation between objects)
            eij = edge_fn(eij || hi || ht)
        2. Compute effects of senders on receivers
            fij = effect_fn(eij || hi || hj)
        3. Aggregate effects to receivers
        4. Update nodes
            hi = node_fn(hi || aggr_effects)
    """
    #-------------------------------------------------------------------
    node_fn: t.Callable
    edge_fn: t.Callable
    effect_fn: t.Callable
    aggregation_fn: t.Callable
    residual: bool
    #-------------------------------------------------------------------

    def __init__(self, node_fn: t.Callable, edge_fn: t.Callable, effect_fn: t.Callable,
                 aggregation_fn: t.Callable = jax.ops.segment_sum, residual: bool = True):

        self.node_fn = node_fn
        self.edge_fn = edge_fn
        self.effect_fn = effect_fn
        self.aggregation_fn = aggregation_fn
        self.residual = residual

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey):

        nodes, edges, receivers, senders, *_ = graph

        # 1. Update edge attributes (relation between nodes)
        edges = evmap(self.edge_fn)(
            cat([edges, nodes[receivers], nodes[senders]])
        )
        edges = edges * graph.active_edges[..., None]
        
        # 2. Compute effects between nodes depending on their relation
        effects = evmap(self.effect_fn)(
            cat([edges, nodes[receivers], nodes[senders]])
        )
        effects = self.aggregation_fn(effects, receivers, nodes.shape[0])
        
        # 3. update node attributes w.r.t effects
        dnodes = evmap(self.node_fn) (
            cat([nodes, effects])
        )
        dnodes = dnodes * graph.active_nodes[..., None]

        if self.residual:
            nodes = nodes + dnodes
        else:
            nodes = dnodes
        
        return graph._replace(nodes=nodes, edges=edges)



