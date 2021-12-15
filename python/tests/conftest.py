import pytest

# -------------------------------------------------------------------------------
# Add --ceed command line argument
# -------------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--ceed", action="store", default='/cpu/self/ref/blocked')


@pytest.fixture(scope='session')
def ceed_resource(request):
    ceed_resource = request.config.option.ceed

    return ceed_resource

# -------------------------------------------------------------------------------
