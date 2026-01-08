from pathlib import Path

import pytest


@pytest.fixture
def test_dir() -> Path:
    """Return the path to the test folder"""
    return Path(__file__).parent


@pytest.fixture
def data_dir(test_dir: Path) -> Path:
    """Return the path to the data folder for tests"""
    return test_dir / "data"
