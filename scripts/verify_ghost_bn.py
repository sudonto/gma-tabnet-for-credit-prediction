"""Check the vectorised ghost batch norm against the reference implementation.

The reference is ``pytorch_tabnet.tab_network.GBN`` (pytorch-tabnet 4.1): it
splits the batch with ``torch.chunk`` and runs a ``BatchNorm1d`` on each chunk in
turn. Agreement of the forward output alone does not establish equivalence, so
for every batch size in the grid, including ones that leave an incomplete last
chunk, this compares

* training-mode outputs,
* gradients with respect to the input, the scale and the shift,
* running mean and running variance after ten training steps,
* eval-mode outputs after those steps,

in float64 (tolerance 1e-9) and float32 (tolerance 5e-6). Deviations are relative,
max|ours - reference| / max(1, max|reference|): a gradient summed over 4,096 rows
is of order 100, and float32 summation order alone moves it by 1e-5 in absolute
terms; 320 sequential running-variance updates likewise drift a few dozen ulps
from the closed form. It exits non-zero on any mismatch, so it can gate a run.
It then times both on the GPU, if present.

    python -m scripts.verify_ghost_bn
"""

from __future__ import annotations

import sys
import time

import torch
from pytorch_tabnet.tab_network import GBN

from src.paper2_credit.tabnet_gated import GhostBatchNorm

FEATURES = 12
MOMENTUM = 0.02
ROW_COUNTS = (4096, 4000, 1000, 513, 300, 257, 129, 64)
VIRTUAL_BATCHES = (128, 256)
TOLERANCE = {torch.float64: 1e-9, torch.float32: 5e-6}


def gap(ours: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative deviation, scaled by the reference's magnitude when it exceeds one."""
    return ((ours - reference).abs().max() / reference.abs().max().clamp_min(1.0)).item()


def compare(n_rows: int, vbs: int, dtype: torch.dtype, device: torch.device) -> dict[str, float]:
    torch.manual_seed(n_rows * 7 + vbs)
    ours = GhostBatchNorm(FEATURES, vbs, MOMENTUM).to(device, dtype).train()
    reference = GBN(FEATURES, virtual_batch_size=vbs, momentum=MOMENTUM).to(device, dtype).train()
    with torch.no_grad():
        scale = torch.randn(FEATURES, dtype=dtype) * 0.2 + 1.0
        shift = torch.randn(FEATURES, dtype=dtype) * 0.2
        for layer_weight, layer_bias in ((ours.weight, ours.bias),
                                         (reference.bn.weight, reference.bn.bias)):
            layer_weight.copy_(scale)
            layer_bias.copy_(shift)

    deviation = {"output": 0.0, "grad_x": 0.0, "grad_weight": 0.0, "grad_bias": 0.0}
    for step in range(10):
        x = (torch.randn(n_rows, FEATURES, dtype=dtype, device=device) * 3 + step).requires_grad_()
        x_ref = x.detach().clone().requires_grad_()
        upstream = torch.randn(n_rows, FEATURES, dtype=dtype, device=device)

        out, out_ref = ours(x), reference(x_ref)
        (out * upstream).sum().backward()
        (out_ref * upstream).sum().backward()

        deviation["output"] = max(deviation["output"], gap(out, out_ref))
        deviation["grad_x"] = max(deviation["grad_x"], gap(x.grad, x_ref.grad))
        deviation["grad_weight"] = max(
            deviation["grad_weight"], gap(ours.weight.grad, reference.bn.weight.grad)
        )
        deviation["grad_bias"] = max(
            deviation["grad_bias"], gap(ours.bias.grad, reference.bn.bias.grad)
        )
        ours.zero_grad()
        reference.zero_grad()

    deviation["running_mean"] = gap(ours.running_mean, reference.bn.running_mean)
    deviation["running_var"] = gap(ours.running_var, reference.bn.running_var)

    ours.eval()
    reference.eval()
    with torch.no_grad():
        x = torch.randn(n_rows, FEATURES, dtype=dtype, device=device) * 3
        deviation["eval_output"] = gap(ours(x), reference(x))
    return deviation


def timing(device: torch.device) -> None:
    x = torch.randn(4096, 64, device=device)
    vectorised = GhostBatchNorm(64, 256).to(device).train()
    reference = GBN(64, virtual_batch_size=256).to(device).train()

    def time_it(fn) -> float:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / 200

    fast, slow = time_it(lambda: vectorised(x)), time_it(lambda: reference(x))
    print(f"\nspeed, 4096 x 64, 16 chunks: vectorised {fast * 1e3:.3f} ms, "
          f"reference {slow * 1e3:.3f} ms, {slow / fast:.1f}x")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    for dtype in (torch.float64, torch.float32):
        tolerance = TOLERANCE[dtype]
        print(f"{str(dtype).replace('torch.', '')}, tolerance {tolerance:g}")
        for vbs in VIRTUAL_BATCHES:
            for n_rows in ROW_COUNTS:
                size, full, tail = GhostBatchNorm.chunk_layout(n_rows, vbs)
                worst = compare(n_rows, vbs, dtype, device)
                bad = {k: v for k, v in worst.items() if v > tolerance}
                failures += bool(bad)
                status = "FAIL" if bad else "OK  "
                print(f"  {status} n={n_rows:<5} vbs={vbs:<4} chunks={full}x{size}"
                      f"{f'+{tail}' if tail else '':<5} max deviation {max(worst.values()):.2e}"
                      + (f"  {bad}" if bad else ""))
    if device.type == "cuda":
        timing(device)
    print("\nall checks passed" if not failures else f"\n{failures} configuration(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
