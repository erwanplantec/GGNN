from src.training._utils import progress_bar_scan
from src.utils._jax_utils import smap

import jax
import jax.numpy as jnp
import jax.random as jr
from evosax import Strategy, EvoParams, ParameterReshaper
import typing as t


def EvosaxTrainer(eval_fn: t.Callable, 
                  strategy: Strategy, 
                  es_params: EvoParams, 
                  params_shaper: ParameterReshaper, 
                  gens: int = 100,
                  progress_bar: bool = True,
                  n_repeats: int = 1,
                  use_vmap: bool = True,
                  var_penalty: float = 0.)->t.Callable:

    """Wrapper for evosax."""

    if n_repeats > 1:
        if use_vmap:
            mapped_eval = jax.vmap(eval_fn, in_axes=(0, None))
        else:
            mapped_eval = smap(eval_fn, 2, (0, ))

        def eval_fn(k, p):
            fits = mapped_eval(jr.split(k, n_repeats), p) #(nrep, pop)"
            avg = jnp.mean(fits, axis=0) #(pop,)
            var = jnp.var(fits, axis=0) #(pop,)
            return avg - var*var_penalty
         
    def evo_step(carry, x):
        key, es_state = carry
        key, ask_key, eval_key = jr.split(key, 3)
        flat_params, es_state = strategy.ask(ask_key, es_state, es_params)
        params = params_shaper.reshape(flat_params)
        fitness = eval_fn(eval_key, params)
        es_state = strategy.tell(flat_params,
                             fitness,
                             es_state,
                             es_params)
        return [key, es_state], [es_state, fitness]

    if progress_bar: evo_step = progress_bar_scan(gens)(evo_step)

    def _train(key: jr.PRNGKey, **init_kws):

        strat_key, evo_key = jr.split(key)
        es_state = strategy.initialize(strat_key, es_params)
        es_state = es_state.replace(**init_kws)
        _, [es_states, fitnesses] = jax.lax.scan(
            evo_step, [evo_key, es_state],
            jnp.arange(gens)
        )
        return es_states, fitnesses

    return _train