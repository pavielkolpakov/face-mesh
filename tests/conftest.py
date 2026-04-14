import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Regenerate golden JSON fixtures instead of asserting.",
    )


@pytest.fixture
def regen_golden(request) -> bool:
    return bool(request.config.getoption("--regen-golden"))
