"""Evaluation of the trained model on a fixed, preprocessed test batch.

    batch, stats = preprocess_batch(generate_icl_batch(...), device)
    evaluator = Evaluator(scalars=[TopKAccuracy(1), LossMetric()],
                          artifacts=[ComposedMatrices()], loss_fn=loss_fn)
    metrics   = evaluator.scalars(model, batch, step)     # dict[str, float]
    artifacts = evaluator.artifacts(model, batch, step)   # {(name, group): (data, type)}

Both calls at the same `step` share one forward pass through `EvalContext`.
"""
from .batch import PreprocessStats, preprocess_batch
from .schedule import evaluation_steps, get_evaluation_times
from .evaluator import Artifacts, EvalContext, Evaluator, Probe, log_artifacts
from .scalars import (BlockVariances, LossMetric, OrderParameters, TargetProbMass, TopKAccuracy,
                      order_parameter_probes)
from .artifacts import AttentionMaps, ComposedMatrices, MProfile, TriggerLogitTable

__all__ = [
    # batch
    "preprocess_batch", "PreprocessStats",
    # schedule
    "evaluation_steps", "get_evaluation_times",
    # machinery
    "EvalContext", "Evaluator", "Probe", "Artifacts", "log_artifacts",
    # scalar probes
    "LossMetric", "TopKAccuracy", "TargetProbMass",
    "OrderParameters", "BlockVariances", "order_parameter_probes",
    # artifact probes
    "ComposedMatrices", "TriggerLogitTable", "AttentionMaps", "MProfile",
]
