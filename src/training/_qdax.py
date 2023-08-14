import jax
import jax.numpy as jnp
import jax.random as jr
import typing as t
from functools import partial
import qdax as qd
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.tasks.arm import arm_scoring_function
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics

default_emitter = None

class QdaxTrainer:

	"""
	Args:

	eval_fn (Callable): Genotype x RNGKey -> Fitness x Descr x ExtraData
	"""
	#-------------------------------------------------------------------
	
	def __init__(self, 
				 eval_fn: t.Callable):
		
		self.emitter = MixingEmitter()
		variation_fn = partial(isoline_variation,
							   iso_sigma=.05,
							   line_sigma=.1)
		emitter = MixingEmitter(
			mutation_fn = lambda x, y: (x, y),
			variation_fn = variation_fn,
			variation_percentage = 1.0, 
			batch_size = batch_size
		)

		metrics_fn = default_qd_metrics

		mapelites = MAPElites(
			scoring_fn = eval_fn
		)

	#-------------------------------------------------------------------

	def __call__(self, key: jr.PRNGKey):
		
		pass