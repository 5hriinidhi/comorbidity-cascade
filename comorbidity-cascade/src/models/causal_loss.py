import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CausalConsistencyLoss(nn.Module):
    """
    Penalizes violations of the causal DAG ordering:
    if a downstream disease has higher predicted probability than its
    upstream cause, the squared violation is penalized.
    """
    def __init__(self, dag, disease_names, lambda_weight=0.15):
        super(CausalConsistencyLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.relu = nn.ReLU()

        # Pre-compute (src_idx, dst_idx, weight) from dag edges
        self.edge_indices = []
        for src, dst, weight in dag.edges:
            src_idx = disease_names.index(src)
            dst_idx = disease_names.index(dst)
            self.edge_indices.append((src_idx, dst_idx, weight))

    def forward(self, y_pred):
        """
        Args:
            y_pred: (batch, 7) tensor of probabilities in [0, 1]
        Returns:
            L_causal: scalar loss
        """
        loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        for src_idx, dst_idx, w in self.edge_indices:
            # violation: downstream prob exceeds upstream prob
            violation = self.relu(y_pred[:, dst_idx] - y_pred[:, src_idx])
            term = w * violation.pow(2).mean()
            loss = loss + term

        return self.lambda_weight * loss


if __name__ == "__main__":
    from src.models.graph_propagation import ComorbidityDAG

    print("--- TESTING CausalConsistencyLoss ---\n")

    config_path = 'config.yaml'
    dag = ComorbidityDAG(config_path)
    disease_names = ["obesity", "t2d", "hypertension", "cad", "ckd", "stroke", "osteoporosis"]

    # ─────────────────────────────────────────────
    # Test 1: Perfect cascade → loss == 0.0
    # ─────────────────────────────────────────────
    loss_fn = CausalConsistencyLoss(dag, disease_names, lambda_weight=0.15)

    # Upstream > downstream for every edge
    batch_size = 16
    y_perfect = torch.zeros(batch_size, 7)
    # obesity=0.9, t2d=0.7, hypertension=0.6, cad=0.5, ckd=0.4, stroke=0.3, osteoporosis=0.2
    y_perfect[:, 0] = 0.9   # obesity
    y_perfect[:, 1] = 0.7   # t2d
    y_perfect[:, 2] = 0.6   # hypertension
    y_perfect[:, 3] = 0.5   # cad
    y_perfect[:, 4] = 0.4   # ckd
    y_perfect[:, 5] = 0.3   # stroke
    y_perfect[:, 6] = 0.2   # osteoporosis

    loss_perfect = loss_fn(y_perfect)
    print(f"Test 1 — Perfect cascade loss: {loss_perfect.item():.6f}")
    assert loss_perfect.item() < 1e-6, f"FAIL: Expected ~0.0, got {loss_perfect.item()}"
    print("  ✅ PASS (loss < 1e-6)\n")

    # ─────────────────────────────────────────────
    # Test 2: Full inversion → loss > 0
    # ─────────────────────────────────────────────
    y_inverted = torch.zeros(batch_size, 7)
    # upstream = 0.1, downstream = 0.9
    y_inverted[:, 0] = 0.1   # obesity (upstream)
    y_inverted[:, 1] = 0.9   # t2d
    y_inverted[:, 2] = 0.9   # hypertension
    y_inverted[:, 3] = 0.9   # cad
    y_inverted[:, 4] = 0.9   # ckd
    y_inverted[:, 5] = 0.9   # stroke
    y_inverted[:, 6] = 0.1   # osteoporosis (no edges)

    loss_inverted = loss_fn(y_inverted)
    print(f"Test 2 — Full inversion loss:  {loss_inverted.item():.6f}")
    assert loss_inverted.item() > 0, f"FAIL: Expected > 0, got {loss_inverted.item()}"
    print("  ✅ PASS (loss > 0)\n")

    # ─────────────────────────────────────────────
    # Test 3: Backward pass works
    # ─────────────────────────────────────────────
    y_grad = torch.randn(batch_size, 7).sigmoid().requires_grad_(True)
    loss_grad = loss_fn(y_grad)
    loss_grad.backward()
    print(f"Test 3 — Backward pass loss:   {loss_grad.item():.6f}")
    assert y_grad.grad is not None, "FAIL: No gradients computed"
    print("  ✅ PASS (gradients flow correctly)\n")

    # ─────────────────────────────────────────────
    # Test 4: Lambda weight scaling
    # ─────────────────────────────────────────────
    loss_fn_2x = CausalConsistencyLoss(dag, disease_names, lambda_weight=0.30)
    loss_1x = loss_fn(y_inverted).item()
    loss_2x = loss_fn_2x(y_inverted).item()
    ratio = loss_2x / loss_1x if loss_1x > 0 else float('inf')
    print(f"Test 4 — Lambda scaling:       1x={loss_1x:.6f}, 2x={loss_2x:.6f}, ratio={ratio:.2f}")
    assert abs(ratio - 2.0) < 1e-4, f"FAIL: Expected ratio ~2.0, got {ratio}"
    print("  ✅ PASS (doubling lambda doubles loss)\n")

    print("CausalConsistencyLoss — All tests PASSED ✅")
