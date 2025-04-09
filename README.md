# Operationally Robust Energy Systems

## Description
This repository contains the code to the paper "TODO". This includes:

* data
* an energy system model in the framework *FINE*
* an energy system model in *gurobipy*
* a framework to generate solutions for individual reference time series, and then cross-evaluate them for other possible time series
* multiple algorithms that are meant to derive solutions that are feasible for all time series at the same time

## Getting started
To get started, execute one of the **main.py** files. Doing so requires some dependencies:

### Installing FINE
To install ETHOS.FINE, visit their [website](https://github.com/FZJ-IEK3-VSA/FINE). Note that *FINE* is only needed for **FINE_Optimization** algorithms, and all **Gurobipy_Optimization** code works without *FINE*.

### Installing Gurobi
You can install *Gurobi* via their [Official Website](https://www.gurobi.com/downloads/). There you can also find information on licenses, including free academic licenses. 

### Installing gurobipy and Other Libaries
Use the package manager of your choice, e.g., "pip3 install gurobipy" for *gurobipy*.

## Generating New Data
TODO add something on the different options
+ FINE ding
evaluate_load_stabilzation
ALG4_load_stabilzation


## Evaluating Results
All results can be found in the **Results** folder, sorted by algorithms. For each algorithm, the results contain the final energy systems generated on the basis of the respective starting years, and their runtimes. 
Results are only given for algorithms that converge, as otherwise neither final ESM nor runtimes are interpretable.

We provide two functions for evaluation: **statistify**, which gives return some statistic key parameters, and **plot_multiple_years** which produces a plot and more in-depth information. See **main.py** for an example of how to use either.

## Sources of Data
This repository also contains data that was provided by third parties. We thank all the people and instutions below for contributing to open access and thereby enabeling our research:

### Wind and PV data: 
We used data from [renewables.ninja](www.renewables.ninja) originally provided by Staffell and Pfenninger in 

*S. Pfenninger and I. Staffell, “Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data” Energy, vol. 114, pp. 1251–1265, 2016.*

*I. Staffell and S. Pfenninger, “Using bias-corrected reanalysis to simulate current and future wind power output,” Energy, vol. 114, pp. 1224–1239, 2016.*

### Geodata 
The basis for geodata is the Nomenclature des Unités territoriales statistiques – NUTS, a classification of the EU. Level 2 of this classification is used in this work, see:

*Council of European Union, “Council regulation (EU) no 1059/2003” 200*

### Load curves
Load curves for tertiary, household, transport and industry sectors were taken from the Forschungsstelle für Energiewirtschaft e. V. (FfE), see:

*M. F. für Energiewirtschaft e. V. (FfE), “Load curves of the tertiary sector – extremos solideu scenario (europe nuts-3),” 2021.*

*M. F. für Energiewirtschaft e. V. (FfE), “Load curves of the household sector – extremos solideu scenario (europe nuts-3),” 2021.*

*M. F. für Energiewirtschaft e. V. (FfE), “Load curves of the transport sector – extremos solideu scenario (europe nuts-3),” 2021.*

*M. F. für Energiewirtschaft e. V. (FfE), “Load curves of the industry sector – extremos solideu scenario (europe nuts-3),” 2021.*

### Regional maximum Capacity Potential
Regional maximum capacity potentials for wind and PV were taken from Risch et al. (2022), see:

*S. Risch, R. Maier, J. Du, N. Pflugradt, P. Stenzel, L. Kotzur, and D. Stolten, “Tool for renewable energy potentials - database,” Apr. 2022.*

*S. Risch, R. Maier, J. Du, N. Pflugradt, P. Stenzel, L. Kotzur, and D. Stolten, “Potentials of Renewable Energy Sources in Germany and the Influence of Land Use Datasets,” Energies, vol. 15, no. 15, 2022*

## Contributing, Contact & Support
We are happy for any interested readers to contact us via [s.kebrich@fz-juelich.de](mailto:s.kebrich@fz-juelich.de) (S. Kebrich) and [engelhardt@combi.rwth-aachen.de](mailto:engelhardt@combi.rwth-aachen.de) (F. Engelhardt). This includes contributions, ideas for collaboration, and feedback. Look forward to hearing from you!

## Authors and acknowledgment
This code was jointly implemented by Sebastian Kebrich and Felix Engelhardt. 
We thank our co-authors Heidi Heinrichs, Christina Büsing and David Franzmann for their support. 

## License
All code is under a Creative Commons Attribution 4.0 International license. Note that this explicitly does not include the data files used. Please refer to the individual sources when using them. 

## Project status
TODO If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
