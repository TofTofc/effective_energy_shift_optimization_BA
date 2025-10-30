import numpy as np
import effective_energy_shift as efes
import cProfile, pstats

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

delta_time_step = 1

# Run the algorithm

prof = cProfile.Profile()
result = prof.runcall(efes.process_phases,energy_excess, energy_deficit, start_time_phases)

ps = pstats.Stats(prof).sort_stats("cumtime")
ps.sort_stats("cumtime")
ps.print_stats("effective_energy_shift_optimization_BA")


ps.print_callees("process_phases")
ps.print_callees("move_overflow")
ps.print_callees("add_excess_to_phase")
ps.print_callees("remove_excess")
