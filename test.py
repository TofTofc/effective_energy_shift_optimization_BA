import numpy as np
import effective_energy_shift as efes
import cProfile, pstats

array_length = 10

worst_case_scenario = True

if worst_case_scenario:
    power_generation = np.arange(array_length, 0, -1)
    power_demand = np.arange(1, array_length+1)

else:
    seed = 12345
    rng = np.random.default_rng(seed)

    power_generation = rng.integers(0, 10, array_length)
    power_demand = rng.integers(0, 10, array_length)

delta_time_step = 1

# Run the algorithm

prof = cProfile.Profile()
result = prof.runcall(efes.perform_effective_energy_shift,power_generation, power_demand, delta_time_step)
ps = pstats.Stats(prof).sort_stats("cumtime")
ps.sort_stats("cumtime")
ps.print_stats("effective_energy_shift_optimization_BA")

'''
ps.print_callees("perform_effective_energy_shift ")
ps.print_callees("analyse_power_data")
ps.print_callees("process_phases")
ps.print_callees("move_overflow")
ps.print_callees("add_excess_to_phase")
ps.print_callees("remove_excess")
'''

print("input:")
print(result.analysis_results.data_input)



