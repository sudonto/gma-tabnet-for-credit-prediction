"""GMA-TabNet: gated multi-head attentive TabNet.

TabNet (Arik & Pfister, 2021) selects features with a single sparsemax mask per
decision step. GMA-TabNet changes three things and nothing else:

* Each step emits ``H`` sparsemax masks instead of one and mixes them with an
  input-dependent softmax gate. A convex combination of points on the simplex is
  on the simplex, so the mixed mask is still a valid mask and the prior update
  ``P <- P * (gamma - M)`` stays well defined.
* Each head divides its logits by its own learned temperature before the
  sparsemax projection, so how selective a head is becomes a parameter rather
  than a constant.
* A low-rank bilinear cross is added residually to the masked input, with a
  scale initialised at zero so training starts from the vanilla behaviour.

Everything else follows the reference implementation (pytorch-tabnet 4.1)
exactly, so that ``variant="vanilla"`` *is* standard TabNet and the ablation
compares architectures rather than implementations:

* attention masks live over the original columns and are spread over a
  categorical column's embedding through a group matrix;
* the feature transformer shares only its ``Linear`` layers across steps, and
  every GLU layer has its own ghost batch normalisation;
* weights are initialised with the reference's xavier gains and the output
  layer has no bias.

``scripts/verify_tabnet_equivalence.py`` transplants weights from the reference
into this encoder and checks that outputs, masks and gradients agree;
``scripts/verify_ghost_bn.py`` does the same for the ghost batch norm, including
its running statistics.

Two further variants exist only as controls. ``uniform_mix`` averages the heads
with fixed equal weights, which isolates the input-dependent gate. ``mh_wang`` is
our implementation of the multi-head TabNet of Wang and Zhang (2025): standard
TabNet plus two scaled dot-product multi-head attention modules over feature
tokens, concatenated with the encoder output before the classification layer.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.torch_utils import resolve_device, sequential_batches
from src.paper2_credit.hpo import BudgetExhausted

# The reference's epsilon inside log(M); kept identical so the sparsity
# regulariser matches it term for term.
EPSILON = 1e-15

VARIANTS = ("vanilla", "gma", "uniform_mix", "mh_wang")
MULTI_HEAD_VARIANTS = ("gma", "uniform_mix")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016).

    Projects onto the simplex, so unlike softmax it can assign exactly zero to a
    feature. Differentiating through the sort and gather gives the correct
    Jacobian, so no custom backward is needed.
    """
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumulative = sorted_logits.cumsum(dim)

    shape = [1] * logits.dim()
    shape[dim] = -1
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    k = k.view(shape)

    support = k * sorted_logits > (cumulative - 1)
    # At least one: non-finite logits make every comparison false, and a zero
    # support would gather at index -1, a device-side assert that poisons the
    # whole CUDA context rather than failing one model.
    support_size = support.to(logits.dtype).sum(dim=dim, keepdim=True).long().clamp(min=1)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(logits - tau, min=0)


def initialize_non_glu(module: nn.Linear, input_dim: int, output_dim: int) -> None:
    """The reference implementation's initialisation for non-GLU layers."""
    gain = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain)


def initialize_glu(module: nn.Linear, input_dim: int, output_dim: int) -> None:
    """The reference implementation's initialisation for GLU layers."""
    gain = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain)


class GhostBatchNorm(nn.Module):
    """Ghost batch normalisation, equivalent to the reference's ``GBN``.

    The reference splits a batch with ``x.chunk(ceil(N / vbs))`` and runs a
    ``BatchNorm1d`` on each chunk in turn. That costs one kernel launch per chunk
    per layer and dominates the runtime on a latency-bound GPU. Here the equal
    chunks are reshaped to (chunks, rows, features) and normalised in one pass,
    and the shorter tail chunk, if any, separately.

    Equivalence covers more than the forward output. ``BatchNorm1d`` updates its
    running statistics once per chunk, with the unbiased chunk variance, so after
    ``k`` chunks with statistics ``s_1..s_k``

        r_k = (1 - m)^k r_0 + sum_i m (1 - m)^(k - i) s_i,

    which is computed in closed form below. A single leftover row has no variance
    of its own; the reference fails on it, here it is normalised with the pooled
    statistics of the other chunks and contributes nothing to the running ones.
    """

    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    @staticmethod
    def chunk_layout(n_rows: int, virtual_batch_size: int) -> tuple[int, int, int]:
        """(chunk size, number of full chunks, tail rows), as ``torch.chunk`` cuts."""
        n_chunks = max(1, math.ceil(n_rows / virtual_batch_size))
        size = math.ceil(n_rows / n_chunks)
        full = n_rows // size
        return size, full, n_rows - full * size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            normalised = (x - self.running_mean) * torch.rsqrt(self.running_var + self.eps)
            return normalised * self.weight + self.bias

        n_rows, n_features = x.shape
        size, full, tail_rows = self.chunk_layout(n_rows, self.virtual_batch_size)
        if size < 2:
            raise ValueError(f"ghost batch norm needs at least two rows per chunk, got {n_rows}")

        cut = full * size
        main = x[:cut].view(full, size, n_features)
        mean = main.mean(dim=1, keepdim=True)
        var = main.var(dim=1, unbiased=False, keepdim=True)
        pieces = [((main - mean) * torch.rsqrt(var + self.eps)).reshape(cut, n_features)]
        chunk_means = [mean.squeeze(1)]
        chunk_vars = [var.squeeze(1) * size / (size - 1)]

        if tail_rows > 1:
            tail = x[cut:]
            tail_mean = tail.mean(dim=0)
            tail_var = tail.var(dim=0, unbiased=False)
            pieces.append((tail - tail_mean) * torch.rsqrt(tail_var + self.eps))
            chunk_means.append(tail_mean.unsqueeze(0))
            chunk_vars.append((tail_var * tail_rows / (tail_rows - 1)).unsqueeze(0))
        elif tail_rows == 1:
            pooled_mean = mean.mean(dim=(0, 1))
            pooled_var = var.mean(dim=(0, 1))
            pieces.append((x[cut:] - pooled_mean) * torch.rsqrt(pooled_var + self.eps))

        with torch.no_grad():
            means = torch.cat(chunk_means, dim=0)
            variances = torch.cat(chunk_vars, dim=0)
            k = means.shape[0]
            decay = 1.0 - self.momentum
            exponents = torch.arange(k - 1, -1, -1, device=x.device, dtype=x.dtype)
            weights = (self.momentum * decay ** exponents).unsqueeze(1)
            self.running_mean.mul_(decay ** k).add_((weights * means).sum(dim=0))
            self.running_var.mul_(decay ** k).add_((weights * variances).sum(dim=0))

        return torch.cat(pieces, dim=0) * self.weight + self.bias


class GLULayer(nn.Module):
    """Linear -> ghost batch norm -> gated linear unit."""

    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.output_dim = output_dim
        self.fc = fc if fc is not None else nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)
        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size, momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        return x[:, : self.output_dim] * torch.sigmoid(x[:, self.output_dim :])


class GLUBlock(nn.Module):
    """A run of GLU layers with sqrt(0.5)-scaled residuals.

    The first layer of the first block cannot carry a residual because it changes
    the width; ``first`` marks that block, exactly as the reference does.
    """

    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None,
                 virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.first = first
        self.layers = nn.ModuleList()
        for i in range(n_glu):
            fc = shared_layers[i] if shared_layers is not None else None
            self.layers.append(
                GLULayer(input_dim if i == 0 else output_dim, output_dim, fc,
                         virtual_batch_size, momentum)
            )
        # The reference computes sqrt(0.5) in float32; the same constant here
        # keeps the two networks identical even when run in float64.
        self.scale = float(np.float32(np.sqrt(np.float32(0.5))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        if self.first:
            x = self.layers[0](x)
            start = 1
        for layer in self.layers[start:]:
            x = (x + layer(x)) * self.scale
        return x


class FeatureTransformer(nn.Module):
    """Shared trunk (shared Linear layers, step-specific batch norm) plus independent layers."""

    def __init__(self, input_dim, output_dim, shared_fcs, n_independent,
                 virtual_batch_size, momentum):
        super().__init__()
        if shared_fcs is not None and len(shared_fcs) > 0:
            self.shared = GLUBlock(input_dim, output_dim, n_glu=len(shared_fcs), first=True,
                                   shared_layers=shared_fcs,
                                   virtual_batch_size=virtual_batch_size, momentum=momentum)
            is_first = False
        else:
            self.shared = None
            is_first = True
        self.specifics = (
            GLUBlock(input_dim if is_first else output_dim, output_dim, n_glu=n_independent,
                     first=is_first, virtual_batch_size=virtual_batch_size, momentum=momentum)
            if n_independent > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared is not None:
            x = self.shared(x)
        if self.specifics is not None:
            x = self.specifics(x)
        return x


class AttentiveTransformer(nn.Module):
    """Vanilla TabNet feature selector: one sparsemax mask per step."""

    def __init__(self, n_a, attention_dim, virtual_batch_size, momentum):
        super().__init__()
        self.fc = nn.Linear(n_a, attention_dim, bias=False)
        initialize_non_glu(self.fc, n_a, attention_dim)
        self.bn = GhostBatchNorm(attention_dim, virtual_batch_size, momentum)

    def forward(self, attention: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.bn(self.fc(attention)) * prior)


class MultiHeadAttentiveTransformer(nn.Module):
    """Several sparse masks per step, mixed convexly.

    Each head owns a projection, a ghost batch norm and a learned sparsity
    temperature. With ``gated=True`` the mixing weights are a softmax of the
    attention features, so different applicants can be routed to different heads;
    with ``gated=False`` they are fixed at 1/H, which is the uniform-mixture
    control. Either way the mixture of simplex points is a simplex point.
    """

    # The inverse of softplus(.) + 1e-3 at 1.0, so every head starts at plain
    # sparsemax exactly.
    RAW_TEMPERATURE_INIT = math.log(math.expm1(1.0 - 1e-3))

    def __init__(self, n_a, attention_dim, n_heads, virtual_batch_size, momentum,
                 learn_temperature=True, gated=True):
        super().__init__()
        self.n_heads = n_heads
        self.gated = gated
        self.projections = nn.ModuleList()
        for _ in range(n_heads):
            fc = nn.Linear(n_a, attention_dim, bias=False)
            initialize_non_glu(fc, n_a, attention_dim)
            self.projections.append(fc)
        self.norms = nn.ModuleList(
            [GhostBatchNorm(attention_dim, virtual_batch_size, momentum) for _ in range(n_heads)]
        )
        self.head_gate = nn.Linear(n_a, n_heads) if gated else None
        self.raw_temperature = nn.Parameter(
            torch.full((n_heads,), self.RAW_TEMPERATURE_INIT), requires_grad=learn_temperature
        )

    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + 1e-3

    def forward(self, attention: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        temperature = self.temperatures()
        masks = []
        for head in range(self.n_heads):
            logits = self.norms[head](self.projections[head](attention)) * prior
            masks.append(sparsemax(logits / temperature[head]))
        stacked = torch.stack(masks, dim=1)                                  # (B, H, D)
        if self.head_gate is None:
            return stacked.mean(dim=1)
        weights = F.softmax(self.head_gate(attention), dim=-1)               # (B, H)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class FeatureInteractionGate(nn.Module):
    """Low-rank bilinear cross added residually to the masked input.

    Starts at exactly zero contribution, so enabling it cannot destabilise
    early training.
    """

    def __init__(self, input_dim: int, rank: int = 16):
        super().__init__()
        self.left = nn.Linear(input_dim, rank, bias=False)
        self.right = nn.Linear(input_dim, rank, bias=False)
        self.project = nn.Linear(rank, input_dim, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.project(self.left(x) * self.right(x))


def embedding_dims(cat_dims: list[int], cat_emb_dim: int) -> list[int]:
    """Embedding width per categorical column: min(cat_emb_dim, max(2, (card + 1) // 2))."""
    return [min(cat_emb_dim, max(2, (dim + 1) // 2)) for dim in cat_dims]


class EmbeddingGenerator(nn.Module):
    """Learned embeddings for the categorical columns, numerics passed through.

    Columns keep their original order, a categorical column being replaced by its
    embedding in place, as in the reference. ``group_matrix`` maps an attention
    mask over original columns onto the embedded columns: a numeric column gets
    its mask value, each embedding column of a categorical gets the value divided
    by the embedding width.
    """

    def __init__(self, input_dim: int, cat_idxs: list[int], cat_dims: list[int], cat_emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cat_idxs = list(cat_idxs)
        widths = embedding_dims(list(cat_dims), cat_emb_dim)
        self.embeddings = nn.ModuleList([nn.Embedding(d, w) for d, w in zip(cat_dims, widths)])
        position_of = {index: i for i, index in enumerate(self.cat_idxs)}
        self._layout = []          # per original column: ("num", None) or ("cat", position)
        source = []
        for column in range(input_dim):
            if column in position_of:
                position = position_of[column]
                self._layout.append(("cat", position))
                source += [column] * widths[position]
            else:
                self._layout.append(("num", None))
                source.append(column)
        self.output_dim = len(source)
        source_t = torch.tensor(source, dtype=torch.long)
        self.register_buffer("source_feature", source_t)
        group = torch.zeros(input_dim, self.output_dim)
        counts = torch.bincount(source_t, minlength=input_dim).to(group.dtype)
        group[source_t, torch.arange(self.output_dim)] = 1.0 / counts[source_t]
        self.register_buffer("group_matrix", group)

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Sum per-embedded-column values (last axis) back onto the original columns."""
        shape = values.shape[:-1] + (self.input_dim,)
        out = torch.zeros(shape, dtype=values.dtype, device=values.device)
        return out.index_add_(values.dim() - 1, self.source_feature.to(values.device), values)

    def expand(self, values: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-original-column vector out to the embedded columns."""
        return values.index_select(-1, self.source_feature.to(values.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for column, (kind, position) in enumerate(self._layout):
            if kind == "num":
                pieces.append(x[:, column : column + 1])
            else:
                table = self.embeddings[position]
                codes = x[:, column].long().clamp(min=0, max=table.num_embeddings - 1)
                pieces.append(table(codes))
        return torch.cat(pieces, dim=1)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
@dataclass
class TabNetConfig:
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 4
    gamma: float = 1.4
    n_shared: int = 2
    n_independent: int = 2
    cat_emb_dim: int = 8
    virtual_batch_size: int = 256
    momentum: float = 0.02
    lambda_sparse: float = 1e-3

    # Architecture. "vanilla" is standard TabNet; see the module docstring.
    variant: str = "gma"
    n_heads: int = 4
    learn_temperature: bool = True
    use_interaction_gate: bool = True
    interaction_rank: int = 16
    # Feature-token attention beside the encoder (Wang and Zhang, 2025, as we
    # read them). variant="mh_wang" is vanilla TabNet with tokens; use_tokens
    # adds them to any variant. wang_heads is the head count of both modules,
    # token_modules keeps both or only one of them, and tokens_only drops the
    # TabNet encoder so the output layer reads the tokens alone.
    wang_heads: int = 6
    use_tokens: bool = False
    token_modules: str = "both"
    tokens_only: bool = False

    # Optimisation.
    learning_rate: float = 2e-2
    weight_decay: float = 1e-5
    batch_size: int = 4096
    max_epochs: int = 120
    patience: int = 20
    focal_gamma: float = 1.5
    focal_alpha: float = 0.75
    use_focal: bool = True
    # Without focal loss: cross-entropy with the positive class weighted by the
    # negative/positive ratio, or plain cross-entropy when False.
    class_weighted: bool = True
    seed: int = 0

    # Training recipe of the reference implementation: Adam without weight
    # decay, no learning-rate schedule, gradient norm clipped at 1 (its
    # clip_value default), unweighted cross-entropy, the last incomplete batch
    # dropped. Used only by the implementation check in the
    # TabNet gap decomposition.
    reference_compat: bool = False

    # Self-supervised pretraining: blank a fraction of the columns and ask the
    # encoder to reconstruct them, then fine-tune with labels.
    pretrain_epochs: int = 0
    pretrain_mask_ratio: float = 0.2

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TabNetConfig":
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class TabNetEncoder(nn.Module):
    """Sequential attention over ``attention_dim`` groups of the embedded input."""

    def __init__(self, input_dim: int, group_matrix: torch.Tensor, config: TabNetConfig):
        super().__init__()
        cfg = config
        if cfg.variant not in VARIANTS:
            raise ValueError(f"unknown variant {cfg.variant!r}")
        self.config = cfg
        self.input_dim = input_dim
        self.register_buffer("group_matrix", group_matrix.clone())
        self.attention_dim = group_matrix.shape[0]
        width = cfg.n_d + cfg.n_a

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.shared_fcs = None
        if cfg.n_shared > 0:
            self.shared_fcs = nn.ModuleList(
                [nn.Linear(input_dim if i == 0 else width, 2 * width, bias=False)
                 for i in range(cfg.n_shared)]
            )

        def feature_transformer():
            return FeatureTransformer(input_dim, width, self.shared_fcs, cfg.n_independent,
                                      cfg.virtual_batch_size, cfg.momentum)

        self.initial_transformer = feature_transformer()
        self.step_transformers = nn.ModuleList([feature_transformer() for _ in range(cfg.n_steps)])

        if cfg.variant in MULTI_HEAD_VARIANTS:
            self.attentive = nn.ModuleList(
                [MultiHeadAttentiveTransformer(
                    cfg.n_a, self.attention_dim, cfg.n_heads, cfg.virtual_batch_size,
                    cfg.momentum, cfg.learn_temperature, gated=cfg.variant == "gma")
                 for _ in range(cfg.n_steps)]
            )
        else:
            self.attentive = nn.ModuleList(
                [AttentiveTransformer(cfg.n_a, self.attention_dim, cfg.virtual_batch_size,
                                      cfg.momentum)
                 for _ in range(cfg.n_steps)]
            )

        use_gate = cfg.variant == "gma" and cfg.use_interaction_gate
        self.gates = (
            nn.ModuleList([FeatureInteractionGate(input_dim, cfg.interaction_rank)
                           for _ in range(cfg.n_steps)])
            if use_gate else None
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None):
        """Return the per-step decision outputs, the sparsity loss and the masks.

        ``prior`` is over the attention groups (original columns). Pretraining
        passes one that is zero on the blanked columns, so the selector cannot
        read off what it is asked to reconstruct.
        """
        cfg = self.config
        x = self.initial_bn(x)
        if prior is None:
            prior = torch.ones(x.shape[0], self.attention_dim, device=x.device, dtype=x.dtype)

        attention = self.initial_transformer(x)[:, cfg.n_d :]
        sparsity_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        steps, masks = [], []
        for step in range(cfg.n_steps):
            mask = self.attentive[step](attention, prior)
            masks.append(mask)
            sparsity_loss = sparsity_loss + torch.mean(
                torch.sum(-mask * torch.log(mask + EPSILON), dim=1)
            )
            prior = prior * (cfg.gamma - mask)

            masked = (mask @ self.group_matrix) * x
            if self.gates is not None:
                masked = self.gates[step](masked)

            out = self.step_transformers[step](masked)
            steps.append(F.relu(out[:, : cfg.n_d]))
            attention = out[:, cfg.n_d :]

        return steps, sparsity_loss / cfg.n_steps, masks


class FeatureTokenAttention(nn.Module):
    """The two multi-head attention modules of Wang and Zhang (2025), as we read them.

    Their description fixes h = 6 heads, scaled dot-product attention, two
    modules ("local" and "global") and concatenation with the TabNet encoder; it
    does not say what the tokens are or how the two modules differ. We use one
    token per original column (a per-column linear map of a numeric value, a
    learned embedding of a category), a self-attention module whose mean-pooled
    output is the local view, and a learned query attending over the tokens for
    the global view.
    """

    def __init__(self, input_dim, cat_idxs, cat_dims, n_heads, width, modules: str = "both"):
        super().__init__()
        if modules not in ("both", "local", "global"):
            raise ValueError(f"unknown token modules {modules!r}")
        self.modules_used = modules
        self.d_model = n_heads * max(1, math.ceil(width / n_heads))
        self.cat_position = {index: i for i, index in enumerate(cat_idxs)}
        self.numeric_idxs = [i for i in range(input_dim) if i not in self.cat_position]
        self.cat_idxs = list(cat_idxs)
        self.numeric_weight = nn.Parameter(torch.randn(len(self.numeric_idxs), self.d_model) * 0.1)
        self.numeric_bias = nn.Parameter(torch.zeros(len(self.numeric_idxs), self.d_model))
        self.cat_tables = nn.ModuleList([nn.Embedding(d, self.d_model) for d in cat_dims])
        self.local = (nn.MultiheadAttention(self.d_model, n_heads, batch_first=True)
                      if modules in ("both", "local") else None)
        if modules in ("both", "global"):
            self.global_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.1)
            self.global_ = nn.MultiheadAttention(self.d_model, n_heads, batch_first=True)
        else:
            self.global_query, self.global_ = None, None
        self.norm = nn.LayerNorm(self.d_model)

    @property
    def output_dim(self) -> int:
        return (2 if self.modules_used == "both" else 1) * self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.numeric_idxs:
            values = x[:, self.numeric_idxs].unsqueeze(-1)
            tokens.append(values * self.numeric_weight + self.numeric_bias)
        for position, column in enumerate(self.cat_idxs):
            table = self.cat_tables[position]
            codes = x[:, column].long().clamp(min=0, max=table.num_embeddings - 1)
            tokens.append(table(codes).unsqueeze(1))
        sequence = self.norm(torch.cat(tokens, dim=1))
        views = []
        if self.local is not None:
            local, _ = self.local(sequence, sequence, sequence, need_weights=False)
            views.append(local.mean(dim=1))
        if self.global_ is not None:
            query = self.global_query.expand(x.shape[0], -1, -1)
            global_, _ = self.global_(query, sequence, sequence, need_weights=False)
            views.append(global_.squeeze(1))
        return torch.cat(views, dim=1)


class ReconstructionDecoder(nn.Module):
    """Map the encoder's decision vector back onto the embedded input.

    Used only during self-supervised pretraining. Deliberately small: its job is
    to make the encoder's representation carry enough about the input to fill in
    blanked columns, not to be a good generative model in its own right.
    """

    def __init__(self, output_dim: int, config: TabNetConfig):
        super().__init__()
        self.block = GLUBlock(config.n_d, config.n_d, n_glu=max(1, config.n_independent),
                              first=False, virtual_batch_size=config.virtual_batch_size,
                              momentum=config.momentum)
        self.project = nn.Linear(config.n_d, output_dim)

    def forward(self, decision: torch.Tensor) -> torch.Tensor:
        return self.project(self.block(decision))


class GMATabNet(nn.Module):
    """Embeddings, encoder and a single-logit output layer."""

    def __init__(self, input_dim: int, config: TabNetConfig,
                 cat_idxs: list[int] | None = None, cat_dims: list[int] | None = None):
        super().__init__()
        cat_idxs = list(cat_idxs or [])
        cat_dims = list(cat_dims or [])
        self.config = config
        self.embedder = EmbeddingGenerator(input_dim, cat_idxs, cat_dims, config.cat_emb_dim)
        has_tokens = config.variant == "mh_wang" or config.use_tokens or config.tokens_only
        self.encoder = (None if config.tokens_only else
                        TabNetEncoder(self.embedder.output_dim, self.embedder.group_matrix, config))
        self.tokens = (
            FeatureTokenAttention(input_dim, cat_idxs, cat_dims, config.wang_heads, config.n_d,
                                  config.token_modules)
            if has_tokens else None
        )
        head_input = (0 if self.encoder is None else config.n_d) + (
            self.tokens.output_dim if self.tokens is not None else 0)
        # One logit instead of the reference's two: sigmoid(w . d) equals the
        # reference's softmax probability when w is the difference of its rows.
        self.head = nn.Linear(head_input, 1, bias=False)
        initialize_non_glu(self.head, head_input, 1)

    def forward_embedded(self, embedded: torch.Tensor, raw: torch.Tensor | None = None):
        if self.encoder is None:
            # Tokens only: no decision steps, no masks, no sparsity penalty.
            zero = torch.zeros((), device=embedded.device, dtype=embedded.dtype)
            return self.head(self.tokens(raw)).squeeze(-1), zero, [], []
        steps, sparsity_loss, masks = self.encoder(embedded)
        decision = torch.stack(steps, dim=0).sum(dim=0)
        if self.tokens is not None:
            decision = torch.cat([decision, self.tokens(raw)], dim=1)
        return self.head(decision).squeeze(-1), sparsity_loss, masks, steps

    def forward(self, x: torch.Tensor):
        logits, sparsity_loss, masks, _ = self.forward_embedded(self.embedder(x), x)
        return logits, sparsity_loss, masks

    def decision(self, x: torch.Tensor) -> torch.Tensor:
        """The representation the output layer reads, which stacking consumes."""
        if self.encoder is None:
            return self.tokens(x)
        steps, _, _ = self.encoder(self.embedder(x))
        decision = torch.stack(steps, dim=0).sum(dim=0)
        if self.tokens is not None:
            decision = torch.cat([decision, self.tokens(x)], dim=1)
        return decision

    def reconstruct(self, x, keep_features, decoder):
        """Encode the partially blanked input and predict the embedded original.

        ``keep_features`` is one on the original columns the encoder may see. It
        is the initial prior over the attention groups and, broadcast to the
        embedded columns, the blanking mask. The target is detached so the
        embedder cannot lower the loss by flattening its own output.
        """
        embedded = self.embedder(x)
        keep = self.embedder.expand(keep_features)
        steps, sparsity_loss, _ = self.encoder(embedded * keep, prior=keep_features)
        decision = torch.stack(steps, dim=0).sum(dim=0)
        return decoder(decision), embedded.detach(), sparsity_loss

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-applicant attention attributions over the original columns.

        Returns (weighted, mass). ``weighted`` is TabNet's own explanation: each
        step's mask weighted by that step's decision magnitude, normalised to sum
        to one per applicant. ``mass`` is the unweighted mask sum, also
        normalised. Both describe which columns the network routed through its
        decision steps, not how much a column moved the prediction.
        """
        _, _, masks, steps = self.forward_embedded(self.embedder(x), x)
        weighted = torch.zeros_like(masks[0])
        mass = torch.zeros_like(masks[0])
        for mask, out in zip(masks, steps):
            weighted = weighted + mask * out.sum(dim=1, keepdim=True)
            mass = mass + mask
        weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weighted, mass

    @torch.no_grad()
    def feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Mean decision-weighted attention per original column."""
        weighted, _ = self.explain(x)
        return weighted.mean(dim=0)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def parameter_count(config: TabNetConfig, input_dim: int, cat_idxs, cat_dims) -> int:
    """Trainable parameters of the network ``config`` would build."""
    return count_parameters(GMATabNet(input_dim, config, cat_idxs, cat_dims))


def match_capacity(target: TabNetConfig, input_dim: int, cat_idxs, cat_dims,
                   widths: range = range(4, 257)) -> tuple[TabNetConfig, int, int]:
    """A vanilla configuration whose parameter count is closest to ``target``'s.

    Only the width ``n_d = n_a`` is changed; depth, steps and every training
    setting stay as in ``target``. Returns (config, its count, the target count).
    """
    goal = parameter_count(target, input_dim, cat_idxs, cat_dims)
    best, best_count = None, None
    for width in widths:
        candidate = replace(target, variant="vanilla", n_d=width, n_a=width,
                            use_tokens=False, tokens_only=False)
        count = parameter_count(candidate, input_dim, cat_idxs, cat_dims)
        if best_count is None or abs(count - goal) < abs(best_count - goal):
            best, best_count = candidate, count
        if count > goal and best_count is not None and best_count != count:
            break
    return best, best_count, goal


# --------------------------------------------------------------------------- #
# sklearn-style trainer
# --------------------------------------------------------------------------- #
def focal_loss(logits, targets, gamma: float, alpha: float) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # Clamped away from zero: for gamma < 1 the derivative of x^gamma is infinite
    # at x = 0, which a saturated float32 sigmoid reaches, and one such row turns
    # every gradient into NaN.
    return (alpha_t * (1 - p_t).clamp_min(1e-6).pow(gamma) * bce).mean()


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise BudgetExhausted


class GMATabNetClassifier:
    """Fit/predict wrapper so the network drops into the same harness as the baselines."""

    def __init__(self, config: TabNetConfig | None = None, device=None) -> None:
        self.config = config or TabNetConfig()
        self.device = resolve_device(device)
        self.model: GMATabNet | None = None
        self.history: list[dict] = []
        self.best_val_auc: float = 0.0
        self.epochs_run: int = 0
        self.fitted_config = self.config
        self.pretrain_loss: float | None = None
        self.diverged = False
        self.input_dim: int | None = None
        self.cat_idxs: list[int] = []
        self.cat_dims: list[int] = []

    def _fit_config(self, n_train: int) -> TabNetConfig:
        """Clamp the batch sizes to something the training set can support.

        German Credit trains on 680 rows, so the default batch of 4096 is the
        whole partition: one gradient step per epoch, and the network learns
        nothing. The batch is capped at the power of two nearest a tenth of the
        training rows (at least 128) and the virtual batch at a quarter of the
        batch (at least 32); a smaller configured value is kept.
        """
        target = 2 ** int(np.log2(max(n_train // 10, 128)))
        batch_size = int(min(self.config.batch_size, max(128, target)))
        virtual_batch_size = int(min(self.config.virtual_batch_size, max(32, batch_size // 4)))
        if (batch_size, virtual_batch_size) == (
            self.config.batch_size, self.config.virtual_batch_size
        ):
            return self.config
        return replace(self.config, batch_size=batch_size, virtual_batch_size=virtual_batch_size)

    def _build(self, input_dim, cat_idxs, cat_dims, cfg) -> None:
        self.input_dim = int(input_dim)
        self.cat_idxs = [int(i) for i in (cat_idxs or [])]
        self.cat_dims = [int(d) for d in (cat_dims or [])]
        self.model = GMATabNet(self.input_dim, cfg, self.cat_idxs, self.cat_dims).to(self.device)

    def fit(self, X_train, y_train, X_val, y_val, cat_idxs=None, cat_dims=None,
            deadline: float | None = None):
        """Train with early stopping on validation ROC-AUC, restoring the best epoch.

        ``deadline`` is an absolute ``time.perf_counter()`` value. When it passes,
        mid-epoch or not, ``BudgetExhausted`` is raised and the evaluation is
        abandoned: a search under a wall-clock budget never runs past it.
        """
        from sklearn.metrics import roc_auc_score

        cfg = self._fit_config(len(y_train))
        self.fitted_config = cfg
        torch.manual_seed(cfg.seed)
        self._build(X_train.shape[1], cat_idxs, cat_dims, cfg)

        if cfg.reference_compat:
            optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
            scheduler = None
        else:
            optimiser = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.max_epochs)

        X_t = torch.as_tensor(np.ascontiguousarray(X_train, dtype=np.float32), device=self.device)
        y_t = torch.as_tensor(np.ascontiguousarray(y_train, dtype=np.float32), device=self.device)
        X_v = torch.as_tensor(np.ascontiguousarray(X_val, dtype=np.float32), device=self.device)

        use_focal = cfg.use_focal and not cfg.reference_compat
        weighted = cfg.class_weighted and not cfg.reference_compat
        pos_weight = None
        if weighted:
            pos_weight = torch.tensor(
                [(y_train == 0).sum() / max(1, (y_train == 1).sum())],
                device=self.device, dtype=torch.float32,
            )
        generator = torch.Generator(device=self.device.type).manual_seed(cfg.seed)

        if cfg.pretrain_epochs > 0:
            self.pretrain_loss = self._pretrain(X_t, cfg, generator, deadline)

        best, best_state, stale = -np.inf, None, 0
        self.diverged = False
        for epoch in range(cfg.max_epochs):
            self.model.train()
            order = torch.randperm(len(X_t), generator=generator, device=self.device)
            epoch_loss, n_batches = 0.0, 0
            for start, stop in sequential_batches(len(X_t), cfg.batch_size):
                idx = order[start:stop]
                if len(idx) < 2 or (cfg.reference_compat and len(idx) < cfg.batch_size
                                    and n_batches > 0):
                    continue
                logits, sparsity, _ = self.model(X_t[idx])
                target = y_t[idx]
                if use_focal:
                    loss = focal_loss(logits, target, cfg.focal_gamma, cfg.focal_alpha)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = loss + cfg.lambda_sparse * sparsity
                if not torch.isfinite(loss):
                    # A configuration that diverges (a learning rate too high
                    # for its width, say) stops here and keeps its best epoch;
                    # in a search it then simply scores badly.
                    self.diverged = True
                    break

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0 if cfg.reference_compat else 5.0
                )
                optimiser.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
                if n_batches % 20 == 0:
                    _check_deadline(deadline)
            if self.diverged:
                break
            if scheduler is not None:
                scheduler.step()
            _check_deadline(deadline)

            val_auc = float(roc_auc_score(y_val, self._scores(X_v)))
            self.history.append(
                {"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "val_auc": val_auc}
            )
            if val_auc > best + 1e-5:
                best, stale = val_auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_auc = float(best)
        self.epochs_run = len(self.history)
        return self

    def _pretrain(self, X_t, cfg, generator, deadline=None) -> float:
        """Fill in blanked columns, so the encoder starts fine-tuning informed.

        Only the training partition's features are used. Reconstruction error is
        divided by each column's standard deviation, as in the TabNet paper.
        """
        assert self.model is not None
        decoder = ReconstructionDecoder(self.model.embedder.output_dim, cfg).to(self.device)
        parameters = list(self.model.embedder.parameters())
        parameters += list(self.model.encoder.parameters()) + list(decoder.parameters())
        optimiser = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        self.model.train()
        decoder.train()
        last = float("nan")
        for _ in range(cfg.pretrain_epochs):
            order = torch.randperm(len(X_t), generator=generator, device=self.device)
            total, n_batches = 0.0, 0
            for start, stop in sequential_batches(len(X_t), cfg.batch_size):
                idx = order[start:stop]
                if len(idx) < 2:
                    continue
                keep_features = (
                    torch.rand(len(idx), self.model.embedder.input_dim,
                               generator=generator, device=self.device)
                    > cfg.pretrain_mask_ratio
                ).float()
                predicted, target, sparsity = self.model.reconstruct(X_t[idx], keep_features, decoder)

                blanked = 1.0 - self.model.embedder.expand(keep_features)
                spread = target.std(dim=0, keepdim=True).clamp_min(1e-3)
                error = ((predicted - target) / spread).pow(2) * blanked
                loss = error.sum() / blanked.sum().clamp_min(1.0) + cfg.lambda_sparse * sparsity

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 5.0)
                optimiser.step()
                total += float(loss.detach())
                n_batches += 1
            last = total / max(n_batches, 1)
            _check_deadline(deadline)

        # Pretraining presents a blanked input, so the normalisation statistics
        # it accumulated describe a distribution fine-tuning never sees. The
        # learned weights are what pretraining was for; the statistics are not.
        for module in self.model.modules():
            if isinstance(module, GhostBatchNorm):
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
        return last

    # ------------------------------------------------------------------ #
    def _tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def _scores(self, X_t: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
        assert self.model is not None
        was_training = self.model.training
        self.model.eval()
        chunks = []
        for start, stop in sequential_batches(len(X_t), batch_size):
            logits, _, _ = self.model(X_t[start:stop])
            chunks.append(np.nan_to_num(torch.sigmoid(logits).cpu().numpy(), nan=0.5))
        self.model.train(was_training)
        return np.concatenate(chunks)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(self._tensor(X))
        return np.column_stack([1 - scores, scores])

    @torch.no_grad()
    def transform(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """The decision representation, which the stacking layer consumes."""
        assert self.model is not None
        self.model.eval()
        X_t = self._tensor(X)
        chunks = [self.model.decision(X_t[a:b]).cpu().numpy()
                  for a, b in sequential_batches(len(X_t), batch_size)]
        return np.vstack(chunks).astype(np.float32)

    @torch.no_grad()
    def explain(self, X: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Per-applicant (decision-weighted, unweighted) attention shares."""
        assert self.model is not None
        self.model.eval()
        X_t = self._tensor(X)
        weighted, mass = [], []
        for a, b in sequential_batches(len(X_t), batch_size):
            w, m = self.model.explain(X_t[a:b])
            weighted.append(w.cpu().numpy())
            mass.append(m.cpu().numpy())
        return np.vstack(weighted), np.vstack(mass)

    def integrated_gradients(self, X: np.ndarray, baseline_row: np.ndarray,
                             steps: int = 32, batch_size: int = 512) -> np.ndarray:
        """Integrated gradients of the logit, per applicant and original column.

        Computed on the embedded input, since categorical codes are not
        differentiable, along the straight line from the embedding of
        ``baseline_row`` (a reference applicant) to each applicant's embedding,
        with a ``steps``-point Riemann sum; per-embedded-column attributions are
        summed back onto the original columns.
        """
        assert self.model is not None
        self.model.eval()
        X_t = self._tensor(X)
        base_raw = self._tensor(np.asarray(baseline_row)[None, :])
        results = []
        alphas = (torch.arange(steps, device=self.device, dtype=torch.float32) + 0.5) / steps
        for a, b in sequential_batches(len(X_t), batch_size):
            raw = X_t[a:b]
            with torch.no_grad():
                embedded = self.model.embedder(raw)
                base = self.model.embedder(base_raw).expand_as(embedded)
            total = torch.zeros_like(embedded)
            for alpha in alphas:
                point = (base + alpha * (embedded - base)).detach().requires_grad_(True)
                logits, _, _, _ = self.model.forward_embedded(point, raw)
                grad, = torch.autograd.grad(logits.sum(), point)
                total += grad
            attribution = (embedded - base) * total / steps
            results.append(self.model.embedder.reduce(attribution).detach().cpu().numpy())
        return np.vstack(results)

    @torch.no_grad()
    def learned_temperatures(self) -> dict[str, list[float]]:
        """The sparsity temperature each head settled on."""
        if self.model is None or self.fitted_config.variant not in MULTI_HEAD_VARIANTS:
            return {}
        return {
            f"step_{step}": [round(float(t), 4) for t in block.temperatures()]
            for step, block in enumerate(self.model.encoder.attentive)
        }

    def parameter_count(self) -> int:
        assert self.model is not None
        return count_parameters(self.model)

    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        assert self.model is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.fitted_config.as_dict(),
                "input_dim": self.input_dim,
                "cat_idxs": self.cat_idxs,
                "cat_dims": self.cat_dims,
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "best_val_auc": self.best_val_auc,
                "epochs_run": self.epochs_run,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device=None) -> "GMATabNetClassifier":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = TabNetConfig.from_dict(payload["config"])
        model = cls(config, device=device)
        model.fitted_config = config
        model._build(payload["input_dim"], payload["cat_idxs"], payload["cat_dims"], config)
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        model.best_val_auc = payload.get("best_val_auc", 0.0)
        model.epochs_run = payload.get("epochs_run", 0)
        return model
