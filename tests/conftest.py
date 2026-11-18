
import pytest

from avsync.models import ensure_models


@pytest.fixture(scope="session", autouse=True)
def _models():
    ensure_models()