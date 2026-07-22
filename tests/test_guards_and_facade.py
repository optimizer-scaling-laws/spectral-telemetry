import pytest


def test_facade_numpy_names_work_without_touching_torch():
    from spectral_telemetry.telemetry import fit_power_law, soft_rank

    assert soft_rank([1.0, 1.0]) == pytest.approx(2.0, abs=1e-6)
    assert abs(fit_power_law([1, 2, 4], [1, 4, 16])["beta"] - 2.0) < 1e-8


def test_package_root_lazy_getattr():
    import spectral_telemetry as st

    assert st.__version__ == "0.1.0"
    assert st.hard_rank([1.0, 1.0, 1.0]) == pytest.approx(3.0, abs=1e-6)


def test_guard_raises_on_dtensor():
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor
    except Exception:
        pytest.skip("DTensor API unavailable")
    from spectral_telemetry.torch_backend.guards import ensure_full_dim_activation

    ensure_full_dim_activation(torch.randn(3, 4))  # plain tensor passes

    if not dist.is_initialized():
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    dt = distribute_tensor(torch.randn(6, 8), mesh, [Shard(1)])
    with pytest.raises(NotImplementedError, match="DTensor|full hidden"):
        ensure_full_dim_activation(dt, context="test")
