ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

.PHONY: install download data features train test all

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# Requires a Kaggle API token and accepted competition rules.
download:
	$(PYTHON) -m kaggle competitions download -c home-credit-default-risk -p data/raw
	$(PYTHON) -c "import zipfile; zipfile.ZipFile('data/raw/home-credit-default-risk.zip').extractall('data/raw')"

data:
	$(PYTHON) -m risklens.data.split

features:
	$(PYTHON) -m risklens.features.build

train:
	$(PYTHON) -m risklens.train

test:
	$(PYTHON) -m pytest -v

all: data features train test
