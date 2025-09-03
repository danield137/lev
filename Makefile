.PHONY: run test test-live fmt lint install install-dev clean help setup-venv activate

# Virtual environment configuration
VENV_DIR := .venv
VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
PYTHON := python

# Check if we're on Windows or Unix-like system
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
    VENV_PIP := $(VENV_DIR)/Scripts/pip.exe
else
    VENV_PYTHON := $(VENV_DIR)/bin/python
    VENV_PIP := $(VENV_DIR)/bin/pip
endif

# Function to ensure venv exists and is activated
define ensure-venv
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
endef

# Default target
help:
	@echo "Available targets:"
	@echo "  setup-venv  - Create virtual environment if it doesn't exist"
	@echo "  activate    - Setup venv and show activation instructions"
	@echo "  run         - Run the context memory server"
	@echo "  test        - Run tests with pytest"
	@echo "  test-live   - Run live integration tests"
	@echo "  fmt         - Format code with black and sort imports with ruff"
	@echo "  lint        - Lint code with ruff"
	@echo "  install     - Install the package"
	@echo "  install-dev - Install package with development dependencies"
	@echo "  clean       - Clean build artifacts"

# Setup virtual environment
setup-venv:
	$(call ensure-venv)

# Activate virtual environment (setup and show instructions)
activate: setup-venv
	@echo "Virtual environment is ready!"
	@echo ""
	@echo "To activate the virtual environment, run:"
ifeq ($(OS),Windows_NT)
	@echo "  .venv\\Scripts\\Activate.ps1"
	@echo "Or in CMD:"
	@echo "  .venv\\Scripts\\activate.bat"
else
	@echo "  source .venv/bin/activate"
endif
	@echo ""
	@echo "To deactivate later, simply run:"
	@echo "  deactivate"

# Run the context memory server
run: setup-venv
	$(VENV_PYTHON) -m fw_context_server.server

# Run tests
test: setup-venv
	$(VENV_PYTHON) -m pytest -v

# Run live integration tests
test-live: setup-venv
	@echo "Running live integration tests..."
	@echo "Note: These tests require external services to be running"
	@echo ""
	$(VENV_PYTHON) tests/live/lm_studio.py

# Format code with isort, black and ruff
fmt: setup-venv
	$(VENV_PYTHON) -m isort --line-length 120 .
	$(VENV_PYTHON) -m black --line-length 120 .

# Lint code with ruff
lint: setup-venv
	$(VENV_PYTHON) -m ruff check --line-length 120 .

# Install the package
install: setup-venv
	$(VENV_PIP) install .

# Install with development dependencies
install-dev: setup-venv
	$(VENV_PIP) install -e ".[dev]"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
