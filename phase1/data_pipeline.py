"""Conditioning-aware data pipeline.

Each training step produces a ConditionedBatch containing:
  - query: Feedback batch — graphs the model must predict (supervised)
  - conditioning: Feedback batch — k example traces the model reads to
    identify the variant (no gradient flows here during training)
  - variant_name: which variant this batch is from (for logging only)

The conditioning examples are separate graphs from the queries, all
executed with the same algorithm variant.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

from clrs._src import samplers

from phase0.variant_registry import (
    BASE_VARIANTS,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    VariantParams,
)
from phase1.sampler import RelaxationSampler

Feedback = samplers.Feedback


@dataclass
class ConditionedBatch:
    """A training batch with conditioning context.

    Attributes:
        query: Feedback with batch_size=B (graphs to predict).
        conditioning: Feedback with batch_size=k (example traces to read).
            These are shared across all queries in the batch (same variant,
            different graphs from the queries).
        variant_name: algorithm variant name (for logging, not model input).
        variant_id: integer index into the variant list (for diagnostics).
    """
    query: Feedback
    conditioning: Feedback
    variant_name: str
    variant_id: int


class ConditionedDataPipeline:
    """Generates ConditionedBatch objects for training or evaluation.

    For each batch:
    1. Samples a variant uniformly from the variant list
    2. Generates B query graphs with traces (supervised targets)
    3. Generates k conditioning graphs with traces (model reads these)
    4. Returns ConditionedBatch

    Args:
        variant_names: list of variant short names to sample from.
        n: number of graph nodes.
        k: number of conditioning examples per batch.
        batch_size: number of query examples per batch.
        p: ER edge probability.
        weight_range: (low, high) for uniform edge weights.
        seed: random seed.
        randomize_pos: randomize positional encodings.
    """

    def __init__(
        self,
        variant_names: Optional[List[str]] = None,
        n: int = 16,
        k: int = 5,
        batch_size: int = 32,
        p: float = 0.3,
        weight_range: tuple = (0.1, 1.0),
        seed: int = 42,
        randomize_pos: bool = True,
        randomize_k: bool = False,
        k_range: tuple = (1, 2, 3, 5, 8),
        per_example_conditioning: bool = False,
    ):
        if variant_names is None:
            variant_names = list(TRAIN_VARIANTS)

        self.variant_names = variant_names
        self.n = n
        self.k = k
        self.batch_size = batch_size
        self.randomize_k = randomize_k
        self.k_range = k_range
        self.per_example_conditioning = per_example_conditioning
        self.rng = np.random.RandomState(seed)

        # Create a sampler per variant with distinct seeds.
        self.samplers: Dict[str, RelaxationSampler] = {}
        for name in variant_names:
            variant = BASE_VARIANTS[name]
            sampler_seed = self.rng.randint(2**31)
            self.samplers[name] = RelaxationSampler(
                variant=variant, n=n, p=p, weight_range=weight_range,
                seed=sampler_seed, randomize_pos=randomize_pos,
            )

        # Build variant name -> integer ID mapping.
        self._all_variant_names = sorted(BASE_VARIANTS.keys())
        self._variant_to_id = {
            name: i for i, name in enumerate(self._all_variant_names)}

    def next(self, allowed_variants=None) -> ConditionedBatch:
        """Generate one conditioned batch.

        With per_example_conditioning=True and mixed_variants (default):
        each query in the batch can be a DIFFERENT variant. Each query gets
        its own k conditioning examples from its variant. The gradient from
        one batch includes signal from multiple variants simultaneously,
        resolving gradient conflicts between variants that share primitives.

        Args:
            allowed_variants: Optional list of variant names to sample from.
                If None, samples from all training variants.
        """
        from phase1.sampler import _stack_feedbacks

        pool = allowed_variants if allowed_variants is not None else self.variant_names
        k = self.rng.choice(self.k_range) if self.randomize_k else self.k

        if self.per_example_conditioning:
            # Mixed-variant batch: each query is independently sampled.
            query_feedbacks = []
            cond_feedbacks = []
            variant_names_batch = []

            for _ in range(self.batch_size):
                name = pool[self.rng.randint(len(pool))]
                sampler = self.samplers[name]
                query_feedbacks.append(sampler._generate_one())
                # k conditioning examples for THIS query, same variant.
                for _ in range(k):
                    cond_feedbacks.append(sampler._generate_one())
                variant_names_batch.append(name)

            query = _stack_feedbacks(query_feedbacks)
            conditioning = _stack_feedbacks(cond_feedbacks)
            # variant_name is a summary (most common, for logging only).
            name = max(set(variant_names_batch),
                       key=variant_names_batch.count)
        else:
            # Single-variant batch: all queries from the same variant.
            name = pool[self.rng.randint(len(pool))]
            sampler = self.samplers[name]
            query = sampler.next(self.batch_size)
            conditioning = sampler.next(k)

        return ConditionedBatch(
            query=query,
            conditioning=conditioning,
            variant_name=name,
            variant_id=self._variant_to_id[name],
        )

    def __iter__(self) -> Iterator[ConditionedBatch]:
        while True:
            yield self.next()


class EvalPipeline:
    """Generates ConditionedBatch objects for evaluation on specific variants.

    Unlike the training pipeline, this generates fixed conditioning sets
    and iterates over eval examples deterministically.

    Args:
        variant_names: variants to evaluate.
        n: number of graph nodes.
        k: number of conditioning examples.
        num_eval_samples: total eval samples per variant.
        eval_batch_size: batch size for eval.
        p: ER edge probability.
        weight_range: (low, high) for uniform edge weights.
        seed: random seed.
    """

    def __init__(
        self,
        variant_names: Optional[List[str]] = None,
        n: int = 16,
        k: int = 5,
        num_eval_samples: int = 128,
        eval_batch_size: int = 32,
        p: float = 0.3,
        weight_range: tuple = (0.1, 1.0),
        seed: int = 123,
    ):
        if variant_names is None:
            variant_names = list(TEST_VARIANTS)

        self.variant_names = variant_names
        self.n = n
        self.k = k
        self.num_eval_samples = num_eval_samples
        self.eval_batch_size = eval_batch_size

        self.rng = np.random.RandomState(seed)

        self.samplers: Dict[str, RelaxationSampler] = {}
        for name in variant_names:
            variant = BASE_VARIANTS[name]
            sampler_seed = self.rng.randint(2**31)
            self.samplers[name] = RelaxationSampler(
                variant=variant, n=n, p=p, weight_range=weight_range,
                seed=sampler_seed, randomize_pos=False,
            )

        self._all_variant_names = sorted(BASE_VARIANTS.keys())
        self._variant_to_id = {
            name: i for i, name in enumerate(self._all_variant_names)}

    def eval_batches(self, variant_name: str) -> Iterator[ConditionedBatch]:
        """Yield eval batches for a specific variant."""
        sampler = self.samplers[variant_name]
        variant_id = self._variant_to_id[variant_name]

        # Generate fixed conditioning set.
        conditioning = sampler.next(self.k)

        # Generate eval batches.
        n_batches = (self.num_eval_samples + self.eval_batch_size - 1) // \
            self.eval_batch_size
        for _ in range(n_batches):
            query = sampler.next(self.eval_batch_size)
            yield ConditionedBatch(
                query=query,
                conditioning=conditioning,
                variant_name=variant_name,
                variant_id=variant_id,
            )

    def all_eval_batches(self) -> Iterator[ConditionedBatch]:
        """Yield eval batches for all variants."""
        for name in self.variant_names:
            yield from self.eval_batches(name)
