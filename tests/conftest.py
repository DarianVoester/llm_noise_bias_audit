"""Shared test fixtures."""

import pytest

from src.providers import reset_unsupported_cache


@pytest.fixture(autouse=True)
def _fresh_unsupported_settings_cache():
    """Each test starts without remembered 'model refuses this setting' entries."""
    reset_unsupported_cache()
    yield
    reset_unsupported_cache()
