# CoordChemPy

[![PyPI](https://img.shields.io/pypi/v/CoordChemPy)](https://pypi.org/project/CoordChemPy/)
![License](https://img.shields.io/github/license/Miguelsd24/ppchem)

## About CoordChemPy

CoordChemPy is a Python package designed to assist inorganic chemists and chemistry students by providing tools for analysis and modeling of coordination compounds. This package includes :

- Calculation about coordination compounds like electron counting, metal electronic structure
- ...

Some approximations and assumptions in order to yield a fully fonctional chemistry package :
- The ligand database is not exhaustive
- Only classical trasition metals were considered. Lanthanides, actinides and heavy synthetic metals (Rf -> Cn) were excluded
- Coordination complexes with more than two metal centers are not incorporated
- Heterobinuclear complexes are not incorporated and homobinuclear complexes must be symmetric with respect to the two metals

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CoordChemPy.

```bash
pip install coordchempy
```

## Utilisation
### Importation
```python
import coordchempy
```
### Functions list and descriptions
All functions are listed and described in a presentation notebook, in the notebook folder.

## Contributing
Contributors are welcome to suggest improvements at https://github.com/Miguelsd24/coordchempy
