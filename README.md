## Effective Energy Shift (EfES) algorithm for Electic Energy Storage (EES) analysis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AirborneKiwi/effective_energy_shift.git/HEAD?labpath=demo_notebook.ipynb)

This repository contains the code for the Effective Energy Shift (EfES) algorithm for Electric Energy Storage (EES) analysis. 
The algorithm is described in the paper "Effective Energy Shift algorithm for Electric Energy Storage analysis" by J. Fellerer, D. Scharrer, and R. German, submitted to Applied Energy, 2024.

It is implemented in Python and an interactive demonstration of the algorithm can be run in the cloud using Binder by clicking on the "launch binder" badge above.

## Installation

Clone the repository and install the required packages using conda:

```bash
conda env create -f environment.yml -n efes_env
conda activate efes_env
```

## Minimal example

```python
import numpy as np
import effective_energy_shift as efes

# Define the input data
power_generation = np.array([2,3,2,4,3,1,0,0,2,5,6,2,1,0,1,2,3,2,0,0,4,4,4,2])
power_demand = np.array([1,4,1,2,1,2,4,5,0,1,3,1,2,2,1,1,2,3,4,5,2,1,5,1])
delta_time_step = 1.

# Run the algorithm
result = efes.perform_effective_energy_shift(power_generation, power_demand, delta_time_step)

# Print the result
print(result.analysis_results.capacity)
print(result.analysis_results.energy_additional)
print(result.analysis_results.self_sufficiency)
print(result.analysis_results.self_consumption)
```

## License

The code is provided under the MIT license. If you use the code in your research, please cite the paper above.

## Contact

If you have any questions, please contact the authors of the paper via <jonathan.fellerer@fau.de>.

