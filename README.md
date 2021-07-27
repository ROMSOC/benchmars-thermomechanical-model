## Coupled parameterized reduced order modelling of thermo-mechanical phenomena arising in blast furnace ##

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

### 4. Authors and contributors

This code has been developed by [Nirav Vasant Shah] [email](mailto:shah.nirav@sissa.it) under the supervision of [Dr. Michele Girfoglio] [email](mailto:michele.girfoglio@sissa.it), [Dr. Patricia Barral] [email](mailto:patricia.barral@usc.es), [Prof. Peregrina Quintela] [email](mailto:peregrina.quintela@itmati.com), [Prof. Gianluigi Rozza] [email](mailto:gianluigi.rozza@sissa.it) and [Ing. Alejandro Lengomin] [email](mailto:alejandro.lengomin@arcelormittal.com).

### 5. How to cite

	@misc{thermomechanical_coupled_one_way-0.1,
	key          = {ThermoMechanicalCoupled},
	author       = {Shah, N.V. and Girfoglio, M. and Barral, P. and Quintela, P. and Rozza, G. and Lengomin, A.},
	title        = {{ThermoMechanicalCoupled} -- {Coupled parameterized reduced order modelling of thermo-mechanical phenomena arising in blast furnaces} {Version 0.1}},
	month        = June,
	year         = 2021,
	doi          = {10.xxxx/zenodo.xxxxxxx},
	note         = {see also: \url{https://zenodo.org/record/3888145}}
	}

### 6. License

* **FEniCS** and **RBniCS** are freely available under the GNU LGPL, version 3.
* **Matplotlib** only uses BSD compatible code, and its license is based on the PSF license. Non-BSD compatible licenses (e.g., LGPL) are acceptable in matplotlib toolkits.

Accordingly, this code is freely available under the GNU LGPL, version 3.

### 7. Disclaimer
In downloading this SOFTWARE you are deemed to have read and agreed to the following terms: This SOFT- WARE has been designed with an exclusive focus on civil applications. It is not to be used for any illegal, deceptive, misleading or unethical purpose or in any military applications. This includes ANY APPLICATION WHERE THE USE OF THE SOFTWARE MAY RESULT IN DEATH, PERSONAL INJURY OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE. Any redistribution of the software must retain this disclaimer. BY INSTALLING, COPYING, OR OTHERWISE USING THE SOFTWARE, YOU AGREE TO THE TERMS ABOVE. IF YOU DO NOT AGREE TO THESE TERMS, DO NOT INSTALL OR USE THE SOFTWARE.

### 8. Acknowledgments
The ROMSOC project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 765374. This repository reflects the views of the author(s) and does not necessarily reflect the views or policy of the European Commission. The REA cannot be held responsible for any use that may be made of the information this repository contains.
