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
    """
    #-------------------------------------------------------------------
    generator: t.Callable
    get_ratio: t.Callable
    #-------------------------------------------------------------------

    def __init__(self, frequency: float, scale: float=1., concentration: float=1., max_freq: float=.9,
                 threshold: float=.1):
        
        self.generator = partial(jr.weibull_min, concentration=concentration, scale=scale)
        def get_ratio(t, key):
            base_freq=frequency[t % len(frequency)]
            mod = self.generator(key)
            ratio = jnp.clip(base_freq * mod, 0., .9)
            ratio = jnp.where(ratio<threshold, 0., ratio)
            return ratio
        self.get_ratio = get_ratio

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey):

        key_w, key_rm = jr.split(key)
        n_edge = graph.active_edges.sum()
        ratio = self.get_ratio(graph.time, key_w)
        return self._rm_edges(graph, ratio, key_rm)
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
