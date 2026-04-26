"""Shared pytest fixtures and markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end tests that load the MLX model (skipped by default in CI)",
    )
    config.addinivalue_line(
        "markers",
        "requires_mlx: tests that need the mlx_audio package (skipped when absent)",
    )


@pytest.fixture(autouse=True)
def _skip_if_mlx_missing(request):
    """Auto-skip tests marked requires_mlx when mlx_audio can't import.

    CI runners (Linux) have no MLX. This lets the same suite run locally
    on Apple Silicon with MLX and in CI without it.
    """
    if request.node.get_closest_marker("requires_mlx"):
        pytest.importorskip("mlx_audio")
