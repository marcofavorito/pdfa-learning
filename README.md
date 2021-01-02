<h1 align="center">
  <b>PDFAs and learning algorithms</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/pdfa-learning">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/pdfa-learning">
  </a>
  <a href="https://pypi.org/project/pdfa-learning">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pdfa-learning" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/pdfa-learning" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/pdfa-learning">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/pdfa-learning">
  </a>
  <a href="https://github.com/marcofavorito/pdfa-learning/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/marcofavorito/pdfa-learning">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/marcofavorito/pdfa-learning/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/marcofavorito/pdfa-learning/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/marcofavorito/pdfa-learning/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/marcofavorito/pdfa-learning">
    <img alt="codecov" src="https://codecov.io/gh/marcofavorito/pdfa-learning/branch/master/graph/badge.svg?token=FG3ATGP5P5">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="" src="https://img.shields.io/badge/flake8-checked-blueviolet">
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="" src="https://img.shields.io/badge/mypy-checked-blue">
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://www.mkdocs.org/">
    <img alt="" src="https://img.shields.io/badge/docs-mkdocs-9cf">
  </a>
</p>


## Install

To install the package from PyPI:
```
pip install pdfa_learning
```

## Setup

- Make sure you have Python 3.7+ on your platform.
- Install [Poetry](https://python-poetry.org/)
- Clone the repository and enter it:
```
git clone https://github.com/marcofavorito/pdfa-learning.git && cd pdfa-learning
```
- Set up the Python virtual environment:
```
poetry shell && poetry install
```
- Install Graphviz if you want to use the rendering features.


## Quickstart

Please have a look at the `notebooks/` 
to see how to use the code.


## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

pdfa-learning is released under the GNU General Public License v3.0 or later (GPLv3+).

Copyright 2020 Marco Favorito

## Authors

- [Marco Favorito](https://marcofavorito.github.io/)
