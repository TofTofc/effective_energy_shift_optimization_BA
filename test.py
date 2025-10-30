import pstats
import numpy as np
from versions.base_case import effective_energy_shift as efes_base_case
from versions.base_case_simple import effective_energy_shift as efes_base_case_simple
import cProfile


def dicts_equal(a: dict, b:dict) -> bool:

    for k in a.keys():
        v_a = a.get(k)
        v_b = b.get(k)
        if not (v_a == v_b).all():
            return False
    return True


array_length = 1000

start_time_phases = np.arange(array_length)

worst_case_scenario = False

if worst_case_scenario:
    energy_excess = np.arange(array_length, 0, -1)
    energy_deficit = np.arange(1, array_length+1)

else:
    seed = 12345
    rng = np.random.default_rng(seed)

    energy_excess = rng.integers(0, 10, array_length)
    energy_deficit = rng.integers(0, 10, array_length)

# Run the algorithm

prof_base_case_simple = cProfile.Profile()
prof_base_case = cProfile.Profile()

result_dict = prof_base_case_simple.runcall(efes_base_case_simple.process_phases,energy_excess, energy_deficit, start_time_phases)
result_dict_base_case = prof_base_case.runcall(efes_base_case.process_phases,energy_excess, energy_deficit, start_time_phases)

print("Base case simple:")
ps = pstats.Stats(prof_base_case_simple).sort_stats("cumtime")
ps.sort_stats("cumtime")
ps.print_stats("effective_energy_shift_optimization_BA")

print("________________________________________________________________________________________________________________")

print("Base case:")
ps = pstats.Stats(prof_base_case).sort_stats("cumtime")
ps.sort_stats("cumtime")
ps.print_stats("effective_energy_shift_optimization_BA")

print("________________________________________________________________________________________________________________")

print("Dicts are equal:")
print(dicts_equal(result_dict, result_dict_base_case))

"""
ps.print_callees("process_phases")
ps.print_callees("move_overflow")
ps.print_callees("add_excess_to_phase")
ps.print_callees("remove_excess")
"""