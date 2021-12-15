clean:
	rm -rf build __pycache__ .pytest_cache *.so

setup:
	$(PYTHON) setup-qfunctions.py build

TEST_OPTS ?= --ceed /cpu/self/ref/serial
test: setup
	$(PYTHON) -m pytest test-*.py $(TEST_OPTS)

.PHONY: clean setup test
