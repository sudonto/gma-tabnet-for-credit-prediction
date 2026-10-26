"""Check that our vanilla TabNet is the reference TabNet, weight for weight.

Builds ``pytorch_tabnet.tab_network.TabNet`` (pytorch-tabnet 4.1) and our network
with ``variant="vanilla"`` on the same mixed categorical/numeric input, copies
every reference weight and batch-norm statistic into ours, and compares in
float64:

* training-mode and eval-mode probabilities (the reference has two output
  logits, ours one; sigmoid of our logit equals the reference's softmax when our
  output row is the difference of its two rows),
* the sparsity regulariser (the reference returns sum M log M, we return its
  negation, and both enter the loss as + lambda * entropy),
* the feature-level attention masks at every step,
* gradients of the training loss with respect to every transplanted parameter,
* every batch-norm running statistic after a training step.

It then checks that the multi-head variant collapses to the vanilla mask when it
has one head at temperature one, which is the claim the ablation's single-head
arm rests on. Exits non-zero on any mismatch.

    python -m scripts.verify_tabnet_equivalence
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet

from src.paper2_credit.tabnet_gated import GMATabNet, TabNetConfig, embedding_dims

TOLERANCE = 1e-9
N_ROWS, VBS = 300, 64
CAT_DIMS = [5, 3, 12]
N_NUMERIC = 6


def make_input(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    cats = [rng.integers(0, d, N_ROWS) for d in CAT_DIMS]
    # Float32-representable values: the reference casts numeric columns with
    # .float() inside its embedding layer, and real inputs are float32 anyway.
    numeric = (rng.normal(size=(N_ROWS, N_NUMERIC)) * 2 + 1).astype(np.float32)
    return torch.tensor(np.column_stack(cats + [numeric]), dtype=torch.float64)


def copy_bn(ours, theirs) -> None:
    """Our GhostBatchNorm <- the reference GBN's inner BatchNorm1d."""
    with torch.no_grad():
        ours.weight.copy_(theirs.bn.weight)
        ours.bias.copy_(theirs.bn.bias)
        ours.running_mean.copy_(theirs.bn.running_mean)
        ours.running_var.copy_(theirs.bn.running_var)


def copy_glu_block(ours, theirs) -> list[tuple]:
    pairs = []
    for mine, ref in zip(ours.layers, theirs.glu_layers):
        with torch.no_grad():
            mine.fc.weight.copy_(ref.fc.weight)
        copy_bn(mine.bn, ref.bn)
        pairs += [(mine.fc.weight, ref.fc.weight), (mine.bn.weight, ref.bn.bn.weight),
                  (mine.bn.bias, ref.bn.bn.bias)]
    return pairs


def transplant(ours: GMATabNet, reference: TabNet) -> list[tuple]:
    """Copy every weight; return (ours, theirs) parameter pairs for gradient checks."""
    pairs = []
    with torch.no_grad():
        for mine, ref in zip(ours.embedder.embeddings, reference.embedder.embeddings):
            mine.weight.copy_(ref.weight)
            pairs.append((mine.weight, ref.weight))
        ref_encoder = reference.tabnet.encoder
        ours.encoder.initial_bn.load_state_dict(ref_encoder.initial_bn.state_dict())
        pairs += [(ours.encoder.initial_bn.weight, ref_encoder.initial_bn.weight)]

    transformers = [(ours.encoder.initial_transformer, ref_encoder.initial_splitter)]
    transformers += list(zip(ours.encoder.step_transformers, ref_encoder.feat_transformers))
    for mine, ref in transformers:
        pairs += copy_glu_block(mine.shared, ref.shared)
        pairs += copy_glu_block(mine.specifics, ref.specifics)

    for mine, ref in zip(ours.encoder.attentive, ref_encoder.att_transformers):
        with torch.no_grad():
            mine.fc.weight.copy_(ref.fc.weight)
        copy_bn(mine.bn, ref.bn)
        pairs += [(mine.fc.weight, ref.fc.weight)]

    with torch.no_grad():
        final = reference.tabnet.final_mapping.weight
        ours.head.weight.copy_((final[1] - final[0]).unsqueeze(0))
    return pairs


def batch_norm_stats(module) -> list[torch.Tensor]:
    return [b for name, b in module.named_buffers() if "running" in name]


def gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1.0)).item()


def main() -> int:
    # The reference keeps its group matrix and priors as plain tensors that
    # .double() does not reach, so the whole check runs in float64 by default.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = TabNetConfig(variant="vanilla", n_d=8, n_a=8, n_steps=3, gamma=1.3, n_shared=2,
                       n_independent=2, cat_emb_dim=4, virtual_batch_size=VBS, momentum=0.02,
                       lambda_sparse=1e-3)
    cat_idxs = list(range(len(CAT_DIMS)))
    input_dim = len(CAT_DIMS) + N_NUMERIC

    reference = TabNet(
        input_dim, 2, n_d=cfg.n_d, n_a=cfg.n_a, n_steps=cfg.n_steps, gamma=cfg.gamma,
        cat_idxs=cat_idxs, cat_dims=CAT_DIMS,
        cat_emb_dim=embedding_dims(CAT_DIMS, cfg.cat_emb_dim),
        n_independent=cfg.n_independent, n_shared=cfg.n_shared, epsilon=1e-15,
        virtual_batch_size=VBS, momentum=cfg.momentum, mask_type="sparsemax",
        group_attention_matrix=torch.eye(input_dim),
    ).double()
    ours = GMATabNet(input_dim, cfg, cat_idxs, CAT_DIMS).double()
    pairs = transplant(ours, reference)

    results: dict[str, float] = {}
    x = make_input(1)
    y = torch.tensor(np.random.default_rng(2).integers(0, 2, N_ROWS), dtype=torch.float64)

    # Training mode: forward, loss, gradients, then the running statistics.
    ours.train()
    reference.train()
    logits, entropy, masks = ours(x)
    ref_out, ref_m_loss = reference(x)
    results["train probability"] = gap(torch.sigmoid(logits), F.softmax(ref_out, dim=1)[:, 1])
    results["sparsity loss"] = gap(entropy, -ref_m_loss)

    loss = F.binary_cross_entropy_with_logits(logits, y) + cfg.lambda_sparse * entropy
    ref_loss = F.cross_entropy(ref_out, y.long()) - cfg.lambda_sparse * ref_m_loss
    results["training loss"] = gap(loss, ref_loss)
    loss.backward()
    ref_loss.backward()
    results["gradients"] = max(gap(a.grad, b.grad) for a, b in pairs)
    results["running statistics"] = max(
        gap(a, b) for a, b in zip(batch_norm_stats(ours.encoder), batch_norm_stats(reference.tabnet.encoder))
    )

    # Eval mode: probabilities and the feature-level masks at every step.
    ours.eval()
    reference.eval()
    with torch.no_grad():
        x = make_input(3)
        logits, _, masks = ours(x)
        ref_out, _ = reference(x)
        results["eval probability"] = gap(torch.sigmoid(logits), F.softmax(ref_out, dim=1)[:, 1])
        _, ref_masks = reference.forward_masks(x)
        embedded_masks = [m @ ours.encoder.group_matrix for m in masks]
        results["masks"] = max(gap(m, ref_masks[s]) for s, m in enumerate(embedded_masks))

    # One head at temperature one with no interaction gate is vanilla TabNet.
    single = GMATabNet(input_dim, TabNetConfig(**{**cfg.as_dict(), "variant": "gma", "n_heads": 1,
                                                    "use_interaction_gate": False}),
                       cat_idxs, CAT_DIMS).double()
    state = {k: v for k, v in ours.state_dict().items()}
    for step, block in enumerate(single.encoder.attentive):
        state[f"encoder.attentive.{step}.projections.0.weight"] = state.pop(f"encoder.attentive.{step}.fc.weight")
        for buffer in ("weight", "bias", "running_mean", "running_var"):
            state[f"encoder.attentive.{step}.norms.0.{buffer}"] = state.pop(f"encoder.attentive.{step}.bn.{buffer}")
        state[f"encoder.attentive.{step}.head_gate.weight"] = block.head_gate.weight.detach().clone()
        state[f"encoder.attentive.{step}.head_gate.bias"] = block.head_gate.bias.detach().clone()
        state[f"encoder.attentive.{step}.raw_temperature"] = block.raw_temperature.detach().clone()
    single.load_state_dict(state)
    single.eval()
    with torch.no_grad():
        x = make_input(4)
        a, _, _ = ours(x)
        b, _, _ = single(x)
    results["single head = vanilla"] = gap(b, a)

    failures = 0
    for name, value in results.items():
        bad = value > TOLERANCE
        failures += bad
        print(f"  {'FAIL' if bad else 'OK  '} {name:<30} {value:.2e}")
    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
