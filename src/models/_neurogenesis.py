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

class NeuroGenesis(eqx.Module):
    
    """
    Node adding layer
        1. Compute division probability for each node
            pi = prob_fn(hi)
        2. Sample dividing nodes wrt p = {pi}
        3. Add new nodes with random embedding 
        4. Add edges from each dividing nodes to their child
    """
    #-------------------------------------------------------------------
    nincr_fn: t.Callable
    eincr_fn: t.Callable
    prob_fn: t.Callable
    threshold: t.Optional[jax.Array]
    gen_fn: t.Callable
    sigma: float
    max_nodes: int
    max_edges: int
    connect: bool
    #-------------------------------------------------------------------
    
    def __init__(self, prob_fn: t.Callable, max_nodes: int, max_edges: int,
                 connect: bool=True, gen_fn: t.Callable = lambda x: x, 
                 threshold: float = None, sigma: float = 0.):
        
        self.prob_fn = prob_fn
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.gen_fn = gen_fn
        self.sigma = sigma
        self.connect = connect
        self.threshold = threshold
        
        mat_n = incr_matrix(max_nodes)
        def incr_nodes(anodes, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_n, 0., 1.), anodes
            ).at[-1].set(0.)
        self.nincr_fn = incr_nodes
        
        mat_e = incr_matrix(max_edges)
        def incr_edges(aedges, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_e, 0., 1.), aedges
            ).at[-1].set(0.)
        self.eincr_fn = incr_edges

    #-------------------------------------------------------------------
        
    @ejit
    def __call__(self, graph: GGraph, key: jr.PRNGKey):
        
        key, key_div = jr.split(key)
        nodes, edges, rec, send, anodes, aedges, *_ = graph
        n_active = anodes.sum().astype(int)
        n_edges = aedges.sum().astype(int)
        ids = jnp.arange(self.max_nodes)
        eids = jnp.arange(self.max_edges)
        
        allowed = self.max_nodes - n_active - 1
        probs = jax.vmap(self.prob_fn)(nodes)[..., 0] * anodes
        if self.threshold is None:
            divs = jnp.where(jr.uniform(key_div, (nodes.shape[0],)) < probs, 1., 0.)
        else :
            divs = jnp.where(probs>self.threshold, 1., 0.)

        return jax.lax.cond(divs.astype(bool).any(),
                            self.add_nodes,
                            lambda g, *_: g,
                            graph, divs, key)

    #-------------------------------------------------------------------

    def add_nodes(self, graph, divs, key):

        key, key_edges, key_nodes = jr.split(key, 3)

        nodes, edges, rec, send, anodes, aedges, *_ = graph
        n_active = anodes.sum().astype(int)
        n_edges = aedges.sum().astype(int)
        ids = jnp.arange(self.max_nodes)
        eids = jnp.arange(self.max_edges)

        nanodes = self.nincr_fn(anodes, n_divs) #add new active nodes

        trgets = jnp.cumsum(divs) * divs - divs
        trgets = jnp.where(divs, trgets.astype(int), -1) + n_edges * divs.astype(int)
        mask_new_nodes = nanodes * (1-anodes)
        new_nodes = jax.ops.segment_sum(jax.vmap(self.gen_fn)(nodes), trgets, self.max_nodes)
        new_nodes = new_nodes + (jr.normal(key_nodes, nodes.shape) * mask_new_nodes[..., None] * self.sigma)

        if self.connect:
            naedges = self.eincr_fn(aedges, n_divs) #add new active edges
            
            mask_new_edges = naedges * (1-aedges)
            
            nsend = jax.ops.segment_sum(ids, trgets, self.max_edges)
            nsend = (send * (1-mask_new_edges) + nsend).astype(int)
            
            nrec = (jnp.cumsum(mask_new_edges)-1) * mask_new_edges + (n_active * mask_new_edges)
            nrec = (rec * (1-mask_new_edges) + nrec).astype(int)

            new_edges = edges + jr.normal(key_edges, edges.shape) * mask_new_edges[..., None] 

            nrec = jnp.where(naedges, nrec, self.max_nodes-1)
            nsend = jnp.where(naedges, nsend, self.max_nodes-1)
        else:
            nsend = send
            nrec = rec
            new_edges = edges
            naedges = aedges
        
        return graph._replace(nodes=new_nodes,
                              edges=new_edges,
                              active_nodes=nanodes,
                              active_edges=naedges,
                              receivers=nrec,
                              senders=nsend)


#=================================================================================================
#=================================================================================================
#=================================================================================================

class NeuroDegeneracy(eqx.Module):

    #-------------------------------------------------------------------
    prob_fn: t.Callable
    max_nodes: int
    max_edges: int
    #-------------------------------------------------------------------

    def __init__(self, prob_fn: t.Callable, max_nodes: int, max_edges: int):
        
        self.prob_fn = prob_fn
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key:jr.PRNGKey):
        
        key, key_prob = jr.split(key)
        nodes, edges, rec, send, anodes, aedges, *_ = graph
        nids = jnp.arange(self.max_nodes)

        # 1. nodes degen
        probs = jax.vmap(self.prob_fn)(nodes)[:, 0] * anodes
        degens = jr.uniform(key_prob, (self.max_nodes,)) < probs
        degens = degens.at[0].set(False)

        nanodes = anodes * degens.astype(float)
        idxs = jnp.argsort(1.-nanodes)
        nanodes = nanodes[idxs]
        new_nodes = jnp.where(nanodes[:, None], nodes[idxs], 0.)

        # 2. edges degen
        degen_ids = jnp.where(degens, nids, jnp.inf)
        is_rec = ((rec[:, None] - degen_ids[None, :]) == 0).any(axis=-1)
        is_send = ((send[:, None] - degen_ids[None, :])==0).any(axis=-1)
        degens = is_rec | is_send
        naedges = aedges * (1.-degens)
        idxs = jnp.argsort(1.-naedges) #actives first, other after
        naedges = naedges[idxs]
        nrec = rec[idxs]
        nrec = jnp.where(naedges, nrec, self.max_nodes-1)
        nsend = send[idxs]
        nsend = jnp.where(naedges, nsend, self.max_nodes-1)
        new_edges = jnp.where(naedges[:, None], edges[idxs], 0.)

        return graph._replace(nodes        = new_nodes,
                              active_nodes = nanodes,
                              active_edges = naedges,
                              receivers    = nrec,
                              senders      = nsend,
                              edges        = new_edges)





