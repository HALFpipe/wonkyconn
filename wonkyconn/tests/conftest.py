from pathlib import Path

import pytest


@pytest.fixture
def data_path(request: pytest.FixtureRequest) -> Path:
    return request.config.rootpath / "data"
