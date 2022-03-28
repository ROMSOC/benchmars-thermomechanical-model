<img src="documentation/images/ROMSOC_Logo.png" alt="ROMSOC logo"  width="150"/>

## Coupled parameterized reduced order modelling of thermo-mechanical phenomena arising in blast furnace ##
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5171821.svg)](https://doi.org/10.5281/zenodo.5171821) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ROMSOC/benchmars-thermomechanical-model/HEAD?labpath=source/mechanical_model/benchmark_mechanical.ipynb)

### 0. Introduction

This repository contains numerical implementation of the benchmark tests reported in "Coupled parameterized reduced order modelling of thermo-mechanical phenomena arising in blast furnace" (see https://zenodo.org/record/3888145). 

### 1. Prerequisites

We use **python** 3.6.9 as the programming language. In this project we use the libraries :
* **FEniCS** 2019.1.0 (www.fenicsproject.org)
* **RBniCS** 0.1.dev1 (www.rbnicsproject.org)
* **Matplolib** 3.1.2 (www.matplotlib.org)
* **numpy** 1.17.4 (www.numpy.org)

The solutions are stored in **.pvd** format, which can later be viewed with **Paraview** (www.paraview.org).

### 2. Installation

Simply clone the public repository:

```
git clone https://github.com/ROMSOC/benchmark_thermomechanical_model
```

### 3. Running the benchmark cases

Source codes for input data are provided in the folder "source_files". Source codes for running the benchmark are provided in folder "source". After running the benchmark, results are stored in folder "result_files".

Run required .py file e.g. "file_name.py" as,
```
python3 file_name.py
```

### 4. Running Jupyter notebooks
The entire benchmark repository can be executed in a provided Docker container where a full installations of FEniCS, RBniCS, and different Python packages are available. Once you have clone or downloaded this repository, to build the container just type
```bash
docker build -t benchmars-thermomechanical-model . 
```
and for running it locally:
```bash
docker run -it --rm -p 8888:8888 benchmars-thermomechanical-model jupyter-lab --ip=0.0.0.0 --port=8888 --allow-root
```

Alternatively, user-friendly Jupyter Notebooks could be used to run different benchmarks on the cloud. For instance, the benchmark is available at:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ROMSOC/benchmars-thermomechanical-model/HEAD?labpath=source/mechanical_model/benchmark_mechanical.ipynb). Please, notice that mybinder cloud computations are limited to 2GB of RAM memory.

### 5. Authors and contributors

This code has been developed by [Nirav Vasant Shah] [email](mailto:shah.nirav@sissa.it) under the supervision of [Dr. Michele Girfoglio] [email](mailto:michele.girfoglio@sissa.it), [Dr. Patricia Barral] [email](mailto:patricia.barral@usc.es), [Prof. Peregrina Quintela] [email](mailto:peregrina.quintela@itmati.com), [Prof. Gianluigi Rozza] [email](mailto:gianluigi.rozza@sissa.it) and [Ing. Alejandro Lengomin] [email](mailto:alejandro.lengomin@arcelormittal.com).

### 6. How to cite

	@misc{Shah_ThermoMechanicalCoupled__Coupled_2021,
	author = {Shah, Nirav V. and Girfoglio, Michele and Barral, Patricia and Quintela, Peregrina and Rozza, Gianluigi and Lengomin-Pieiga, Alejandro},
	doi = {10.5281/zenodo.5171821},
	title = {{ThermoMechanicalCoupled -- Coupled parameterized reduced order modelling of thermo-mechanical phenomena arising in blast furnaces}},
	url = {https://github.com/ROMSOC/benchmars-thermomechanical-model},
	year = {2021}
	}

### 7. License

* **FEniCS** and **RBniCS** are freely available under the GNU LGPL, version 3.
* **Matplotlib** only uses BSD compatible code, and its license is based on the PSF license. Non-BSD compatible licenses (e.g., LGPL) are acceptable in matplotlib toolkits.

Accordingly, this code is freely available under the GNU LGPL, version 3.

### 8. Disclaimer
In downloading this SOFTWARE you are deemed to have read and agreed to the following terms: This SOFT- WARE has been designed with an exclusive focus on civil applications. It is not to be used for any illegal, deceptive, misleading or unethical purpose or in any military applications. This includes ANY APPLICATION WHERE THE USE OF THE SOFTWARE MAY RESULT IN DEATH, PERSONAL INJURY OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE. Any redistribution of the software must retain this disclaimer. BY INSTALLING, COPYING, OR OTHERWISE USING THE SOFTWARE, YOU AGREE TO THE TERMS ABOVE. IF YOU DO NOT AGREE TO THESE TERMS, DO NOT INSTALL OR USE THE SOFTWARE.

### 9. Acknowledgments
<img src="documentation/images/logos/EU_Flag.png" alt="EU Flag"  width="150"/>

The ROMSOC project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 765374. This repository reflects the views of the author(s) and does not necessarily reflect the views or policy of the European Commission. The REA cannot be held responsible for any use that may be made of the information this repository contains.
