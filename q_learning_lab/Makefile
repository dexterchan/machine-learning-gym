.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/flake8: ## check style with flake8
	flake8 q_learning_lab tests

lint: lint/flake8 ## check style


test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source q_learning_lab setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/q_learning_lab.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ q_learning_lab
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

setup_env:
	mamba install -y -c conda-forge --file requirements.txt
	mamba install -y -c conda-forge --file requirements_jupyter.txt
	mamba install -y -c conda-forge --file requirements_dev.txt
	#python3 -m pip install -r requirements_dev.txt

create_container_image:
	sh scripts/container/build_image.sh

create_container_image_slim:
	sh scripts/container/build_image_slim.sh


training: export PYTHONPATH=$(shell pwd)/src
training: export DATA_DIR=$(shell pwd)/../../crypto_market_data/data
training: 
	python3 -m q_learning_lab -b intraday-market-v0 -c scripts/config/intraday_config.json -i local_intraday

training_local: export PYTHONPATH=$(shell pwd)/src
training_local: export DATA_DIR=$(shell pwd)/data
training_local: 
	python3 -m q_learning_lab -b intraday-market-v0 -c scripts/config/intraday_config_local.json -i local_intraday

# training_long: 
# 	python3 -m q_learning_lab -b cartpole-v1 -c scripts/config/cart_pole_v1_long.json

dummy_run: 
	python3 -m q_learning_lab -b dummy -c scripts/config/cart_pole_v1_long.json


test: export PYTHONPATH=$(shell pwd)/src
test: export DATA_DIR=$(shell pwd)/../../crypto_market_data/data
test:
	echo ${PYTHONPATH}
	pytest tests

test_local: export PYTHONPATH=$(shell pwd)/src
test_local: export DATA_DIR=$(shell pwd)/data
test_local:
	echo ${PYTHONPATH}
	pytest tests

run_jupyter: export PYTHONPATH=$(shell pwd)/src
run_jupyter: export DATA_DIR=$(shell pwd)/../../crypto_market_data/data
run_jupyter:
	jupyter lab
	#jupyter notebook --notebook-dir=notebooks

