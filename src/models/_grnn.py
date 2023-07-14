from src.models._graph import GGraph
from src.models._gnca import GNCA

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import equinox.nn as nn

class LSTM(eqx.Module):

    """
    """
    #-------------------------------------------------------------------
    cell: eqx.Module
    h_size: int
    #-------------------------------------------------------------------

    def __init__(self, input_size: int, hidden_size: int,  **kwargs):

        self.cell = nn.LSTMCell(input_size, hidden_size, **kwargs)
        self.h_size = hidden_size

    #-------------------------------------------------------------------

    def __call__(self, x: jax.Array, h: jax.Array):
        
        h = (h[:self.h_size], h[self.h_size:])
        h = self.cell(x, h)
        h = jnp.concatenate(h, axis=-1)
        return h

    #-------------------------------------------------------------------


class GRNCA(GNCA):

    """
    Same algorithm than GNCA but node_fn is a recurrent unit
    """
    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey):
        
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
        nodes = jax.vmap(self.node_fn)(m, graph.nodes)

        return graph._replace(nodes=nodes, edges=edges)
