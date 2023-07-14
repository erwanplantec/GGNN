from src.models._graph import GGraph
from src.models._synaptogenesis import SynapticDegeneracy

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import equinox.nn as nn
import typing as t
from functools import partial


class WeibullDegeneracy(eqx.Module):

    """
    When called, remove some edges with probability <proba>
    Args:
        - proba (float): probability of doing smthing
        - strength (float): proba that an edge is removed
    """
    #-------------------------------------------------------------------
    frequency: float
    generator: t.Callable
    #-------------------------------------------------------------------

    def __init__(self, frequency: float, scale: float=1., concentration: float=1., max_freq: float=.9):
        
        self.frequency = frequency
        self.generator = partial(jr.weibull_min, concentration=concentration, scale=scale)

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey):

        key_w, key_rm = jr.split(key)
        n_edge = graph.active_edges.sum()
        mod = self.generator(key_w)
        freq = jnp.clip(self.frequency * mod, 0., .9)
        n_rm = freq * n_edge
        proba = n_rm / n_edge
        return self._rm_edges(graph, proba, key_rm)

    #-------------------------------------------------------------------

    def _rm_edges(self, graph: GGraph, proba: float, key: jr.PRNGKey):

        nodes, edges, rec, send, anodes, aedges, *_ = graph

        degens = jr.uniform(key, (edges.shape[0],)) < proba
        degens = degens.astype(float)
        naedges = aedges * (1.-degens)
        idxs = jnp.argsort(1.-naedges) #actives first, other after

        naedges = naedges[idxs] #actives first, other after
        nrec = rec[idxs]
        nrec = jnp.where(naedges, nrec, nodes.shape[0]-1)
        nsend = send[idxs]
        nsend = jnp.where(naedges, nsend, nodes.shape[0]-1)
        new_edges = jnp.where(naedges[:, None], edges[idxs], 0.)

        return graph._replace(active_edges = naedges,
                              senders      = nsend,
                              receivers    = nrec,
                              edges        = new_edges)
