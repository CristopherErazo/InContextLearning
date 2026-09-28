"""The probing machinery: one lazily-evaluated context per step, shared by
every probe.

    ctx = EvalContext(model, batch, step, chunk=256)
    ctx.token_loss        # runs the (chunked) forward pass once, on first access
    ctx.on_target_logits  # derived from ctx.logits_ind, also computed once

A *probe* is any callable with a `name` that takes an `EvalContext`. Scalar
probes return a float (or a dict of floats) that is logged as metrics;
artifact probes return a tensor / object (or a dict of them) that is saved to
disk. The two kinds share the protocol and the context; only the packing of
their results differs, and `Evaluator` does that packing.

`Evaluator` memoizes the context on the step number, so `scalars()` and
`artifacts()` called at the same step cost one forward pass in total.

The forward pass runs over the test batch in chunks of `chunk` sequences and
keeps only what the probes read: the per-position cross-entropy (B, L) and the
logits at induction-possible positions (N, V). Nothing of size (B, L, d) or
(B, L, V) outlives a chunk, so evaluation needs no more memory than a training
step on `chunk` sequences. Probes that want the attention maps use `outputs`,
which is NOT chunked.
"""
from __future__ import annotations

from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

ArtifactKey = tuple[str, str | None]    # (name, group)
ArtifactValue = tuple[Any, str]          # (data, tracklab type: 'tensor' | 'pickle' | 'torch')
Artifacts = dict[ArtifactKey, ArtifactValue]


def split_on_off(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (N, V) logits into on-target (N,) and off-target (N, V-1) logits."""
    on = logits.gather(1, targets[:, None]).squeeze(1)
    vocab = torch.arange(logits.size(1), device=logits.device)
    off = logits[vocab[None, :] != targets[:, None]].view(logits.size(0), -1)
    return on, off


class EvalContext:
    """Lazy view of (model, batch) at one training step.

    Cheap tensors (inputs, targets, the induction mask) are set eagerly. Anything that runs
    the model or multiplies weights is a `cached_property`: computed at most
    once per context and only if some probe asks for it. A context must not
    outlive the weights it was built for; `Evaluator` handles that by keying
    on `step`.
    """

    def __init__(self, model, batch: dict, step: int | None = None, chunk: int | None = None):
        self.model = model
        self.step = step
        self.chunk = chunk  # sequences per forward pass; None = the whole batch at once
        self.device = next(model.parameters()).device

        seq = batch["sequence"].to(self.device)                     # (B, L+1)
        self.input = seq[:, :-1]                                    # (B, L)
        self.target = seq[:, 1:]                                    # (B, L)
        self.K = int(batch["K"])                                    # triggers are the tokens 0..K-1
        self.ind_possible = batch["ind_possible"].to(self.device)   # (B, L)
        if model.pred_mode == "last":
            # the model only emits logits for the final query; keep masks aligned with (B, 1, V)
            self.target = self.target[:, -1:]
            self.ind_possible = self.ind_possible[:, -1:]
        self.ind_not_possible = ~self.ind_possible
        self.all = torch.ones_like(self.ind_possible)

    # ---- model outputs: one chunked forward pass for everything ------------------

    def _no_grad_eval(self, fn):
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                return fn()
        finally:
            self.model.train(was_training)

    @cached_property
    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(per-position cross-entropy (B, Lq), logits at induction positions (N, V)).

        The rows of the second come in `torch.where(ind_possible)` order, so they
        line up with `ind_index`.
        """
        def run():
            B = self.input.size(0)
            step = self.chunk or B
            losses, ind_logits = [], []
            for s in range(0, B, step):
                logits = self.model(self.input[s:s + step])            # (b, Lq, V)
                target = self.target[s:s + step]
                losses.append(F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                              reduction="none").view_as(target))
                ind_logits.append(logits[self.ind_possible[s:s + step]])
            return torch.cat(losses), torch.cat(ind_logits)
        return self._no_grad_eval(run)

    @property
    def token_loss(self) -> torch.Tensor:  # (B, L) or (B, 1): cross-entropy at every position
        return self._forward[0]

    @cached_property
    def outputs(self) -> dict[str, torch.Tensor]:
        """`model.full_output` on the WHOLE test batch at once (attention maps and,
        for the full model, X1, X2, S1, S2). Unchunked: only for probes that need
        the maps, and only at sizes where (B, L, L) and (B, L, d) fit."""
        return self._no_grad_eval(lambda: self.model.full_output(self.input))

    @property
    def attn1(self) -> torch.Tensor:   # (B, L, L)
        return self.outputs["A1"]

    @property
    def attn2(self) -> torch.Tensor:   # (B, L, L) or (B, 1, L)
        return self.outputs["A2"]

    # ---- induction positions ----------------------------------------------------

    @cached_property
    def ind_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch_idx, position_idx) of every induction-possible position, each (N,)."""
        return torch.where(self.ind_possible)

    @property
    def n_ind(self) -> int:
        return int(self.ind_index[0].numel())

    @property
    def logits_ind(self) -> torch.Tensor:  # (N, V)
        return self._forward[1]

    @cached_property
    def target_ind(self) -> torch.Tensor:  # (N,)
        return self.target[self.ind_index]

    @cached_property
    def on_off_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return split_on_off(self.logits_ind, self.target_ind)

    @property
    def on_target_logits(self) -> torch.Tensor:   # (N,)
        return self.on_off_logits[0]

    @property
    def off_target_logits(self) -> torch.Tensor:  # (N, V-1)
        return self.on_off_logits[1]

    # ---- vocabulary ---------------------------------------------------------------

    @property
    def V(self) -> int:
        return self.model.vocab_size

    @cached_property
    def trigger_mask(self) -> torch.Tensor:
        """(V,) bool, on the CPU where the composed matrices live: True on the
        trigger tokens, which are always `0..K-1`."""
        return torch.arange(self.V) < self.K

    # ---- weights ------------------------------------------------------------------

    @cached_property
    def matrices(self) -> dict[str, torch.Tensor]:
        """Composed analysis matrices {"M", "Q", "G"} (on CPU), from the model."""
        return {name: m for (name, _group), m in self.model.get_composed_matrices().items()}

    @cached_property
    def normalized_matrices(self) -> dict[str, torch.Tensor]:
        """{"M", "Q", "G"} divided by sqrt(d), the O(1) objects the order parameters
        average. The reduced model stores them directly (finite at d = inf)."""
        if hasattr(self.model, "normalized_matrices"):
            return self.model.normalized_matrices()
        s = self.model.d_model ** 0.5
        return {name: m / s for name, m in self.matrices.items()}

    @property
    def E(self):    return self.model.embed.E.weight.T      # (d, V)
    @property
    def P(self):    return self.model.embed.P.weight.T      # (d, L)
    @property
    def U(self):    return self.model.unembed.U.weight      # (V, d)
    @property
    def WQK1(self): return self.model.attn1.WQK.weight.T    # (d, d)
    @property
    def WQK2(self): return self.model.attn2.WQK.weight.T    # (d, d)
    @property
    def WOV1(self): return self.model.attn1.WOV.weight      # (d, d)
    @property
    def WOV2(self): return self.model.attn2.WOV.weight      # (d, d)


@runtime_checkable
class Probe(Protocol):
    """Anything with a `name` that maps an EvalContext to a result.

    Scalar probes return `float | dict[str, float]`; a dict is merged into the
    metrics under its own keys. Artifact probes return `Any | dict[str, Any]`
    and may set two optional class attributes: `group` (artifact subfolder,
    default None) and `atype` (tracklab serializer, default 'tensor').
    """
    name: str

    def __call__(self, ctx: EvalContext) -> Any: ...


class Evaluator:
    """Owns the probe lists and the per-step context cache."""

    def __init__(self, scalars: list[Probe] = (), artifacts: list[Probe] = (), chunk: int | None = None):
        self.scalar_probes = list(scalars)
        self.artifact_probes = list(artifacts)
        self.chunk = chunk
        self._ctx: EvalContext | None = None
        for probe in (*self.scalar_probes, *self.artifact_probes):
            if not isinstance(probe, Probe):
                raise TypeError(f"{probe!r} is not a probe: it needs a `name` and must be callable")

    def context(self, model, batch: dict, step: int | None = None) -> EvalContext:
        """The context for `step`, reused if it was already built for the same
        step and model. With `step=None` a fresh context is built on every call."""
        stale = (self._ctx is None or step is None
                 or self._ctx.step != step or self._ctx.model is not model)
        if stale:
            self._ctx = EvalContext(model, batch, step, self.chunk)
        return self._ctx

    def scalars(self, model, batch: dict, step: int | None = None) -> dict[str, float]:
        ctx, out = self.context(model, batch, step), {}
        for probe in self.scalar_probes:
            result = probe(ctx)
            items = result if isinstance(result, dict) else {probe.name: result}
            for name, value in items.items():
                if name in out:
                    raise KeyError(f"metric {name!r} is produced by more than one probe")
                out[name] = float(value)
        return out

    def artifacts(self, model, batch: dict, step: int | None = None) -> Artifacts:
        """{(name, group): (data, type)}: the shape Rewind's `eval_artifacts_fn`
        contract accepts and `log_artifacts` below writes."""
        ctx, out = self.context(model, batch, step), {}
        for probe in self.artifact_probes:
            group = getattr(probe, "group", None)
            atype = getattr(probe, "atype", "tensor")
            result = probe(ctx)
            items = result if isinstance(result, dict) else {probe.name: result}
            for name, data in items.items():
                key = (name, group)
                if key in out:
                    raise KeyError(f"artifact {key!r} is produced by more than one probe")
                if torch.is_tensor(data):
                    data = data.detach().cpu()
                    if atype == "tensor":
                        data = data.numpy()
                out[key] = (data, atype)
        return out


def log_artifacts(run, artifacts: Artifacts, step: int) -> None:
    """Write an `Evaluator.artifacts()` result to a tracklab Run."""
    for (name, group), (data, atype) in artifacts.items():
        run.track_artifact(data, step=step, group=group, name=name, type=atype)
