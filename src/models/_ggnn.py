import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import typing as t
from src.models._graph import GGraph
from src.models._utils import rollout

class GGNN(eqx.Module):

    """
    Growing Graph Neural Network wrapper class
    allows to optimize initialization parameters also

    Args:
        model (eqx.Module): graph update model
        n_init_{nodes, edges} (int): number of init {nodes, edges}
        {node, edge}_featues (int): number of {node, edge} features

    __call__: GGRaph X PRNGKey [X int X bool] -> GGraph [X GGraph] 
    """
    #-------------------------------------------------------------------
    model: eqx.Module
    init_nodes: jnp.ndarray
    n_init_nodes: int
    init_edges: jnp.ndarray
    n_init_edges: int
    #-------------------------------------------------------------------

    def __init__(self, model: eqx.Module, n_init_nodes: int, n_init_edges: int, 
                 node_features: int, edge_features: int, key: jr.PRNGKey):

        self.model = model
        self.init_nodes = jnp.zeros((n_init_nodes, node_features))
        self.n_init_nodes = n_init_nodes
        self.init_edges = jnp.zeros((n_init_edges, edge_features))
        self.n_init_edges = n_init_edges

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey, dev_steps: int=20, 
                 return_traj: bool=False, init: bool=True):

        if init:
            graph = graph._replace(
                nodes = graph.nodes.at[:self.n_init_nodes].set(self.init_nodes),
                edges = graph.edges.at[:self.n_init_edges].set(self.init_edges)
            )

        [graph, _], graphs = rollout(self.model, graph, key, dev_steps)
        if return_traj:
            return graph, graphs
        else:
            return graph
