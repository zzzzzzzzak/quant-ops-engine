import torch
import torch.nn as nn

from src.debugger import GradientHealthMonitor, run_numerical_stress_audit


def test_numerical_stress_audit_all_pass():
    audit = run_numerical_stress_audit()
    assert len(audit) == 4
    for name, report in audit.items():
        assert report["status"] == "PASS", f"Stress test failed for {name}: {report}"


def test_gradient_health_monitor_detects_nan_and_exploding_gradients():
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))

    # 1. Healthy backward pass
    x = torch.randn(4, 16)
    with GradientHealthMonitor(model, max_grad_norm=100.0) as monitor:
        loss = model(x).sum()
        loss.backward()
    assert monitor.is_healthy is True

    model.zero_grad()

    # 2. Exploding gradient pass
    x_huge = torch.randn(4, 16) * 1e5
    with GradientHealthMonitor(model, max_grad_norm=50.0) as monitor_exp:
        loss = model(x_huge).sum()
        loss.backward()
    assert monitor_exp.is_healthy is False
    assert any("Exploding" in a.issue for a in monitor_exp.alerts)

    model.zero_grad()

    # 3. NaN gradient pass
    x_nan = torch.full((4, 16), float("nan"))
    with GradientHealthMonitor(model, max_grad_norm=100.0) as monitor_nan:
        loss = model(x_nan).sum()
        loss.backward()
    assert monitor_nan.is_healthy is False
    assert any("NaN" in a.issue for a in monitor_nan.alerts)
