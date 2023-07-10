from src.models import GGraph
from src.metrics import *

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import host_callback as hcb

import typing as t


def MetricTask(statics: t.Collection, init_graph: GGraph, metrics: list,
               targets: jnp.array, grow_iters: int = 20, apply_all_timesteps: bool = True,
               weights: jnp.array=None):
    
    metric_fn = GraphMetrics(metrics)
    if apply_all_timesteps and weights is None:
        weights = jnp.ones((grow_iters,))
    if set(metrics).issubset(set(jax_graph_metrics)):
        @partial(jax.vmap, in_axes=(None, 0))
        def _eval(key, params):
            model = eqx.combine(params, statics)
            graph, graphs = model(init_graph, key, grow_iters, True)
            if apply_all_timesteps:
                ms = jax.vmap(metric_fn)(graphs) #(steps, n_metrics)
                fit = jnp.sum((ms-targets[None, ...])**2, axis=-1) * weights #(steps,)
                fit = jnp.mean(fit)
            else:
                m = metric_fn(graph)
                fit = jnp.mean((m-targets)**2)
            return fit
        return _eval
    
    def callback(graph):
        m = metric_fn(graph)
        return m
    n=len(metrics)
    def sub_eval(key, params):
        model = eqx.combine(params, statics)
        graph, graphs = model(init_graph, key, grow_iters, return_traj=True)
        metrics = hcb.call(callback, graph, result_shape=jnp.ones((n,)))
        fit = jnp.mean((metrics - targets)**2)
        return key, fit
    
    def _eval(key, params):
        _, fits = jax.lax.scan(sub_eval, key, params)
        return fits
    
    return _eval