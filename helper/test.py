import cProfile
import pstats
from helper.compare_methodes import is_equal
from helper.extract_results import extract_results
from helper.visualizer import visualize
from main import init, import_version

delim = "-"*100


def submethod_analysis(version_name, worst_case_scenario = False, phase_count = 1000000):

    module = import_version(version_name)

    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)

    energy_excess_lists_fake, energy_deficit_lists_fake, start_time_phases_fake = init(worst_case_scenario, 125, 10, 1)

    # Fake run to compile
    module.process_phases(energy_excess_lists_fake[0], energy_deficit_lists_fake[0], start_time_phases_fake)

    pr = cProfile.Profile()

    pr.runcall(module.process_phases, energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

    ps = pstats.Stats(pr)
    ps.strip_dirs()
    ps.sort_stats("tottime")
    ps.print_stats()

def test_versions(version_name_a, version_name_b, start = 10, end = 1000, repetitions_count = 25, worst_case_scenario = False):

    module_a = import_version(version_name_a)
    module_b = import_version(version_name_b)

    for i in range(start, end):

        print(i)

        energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, i, repetitions_count)

        for idx in range(repetitions_count):

            phases_list_a = module_a.process_phases(energy_excess_lists[idx], energy_deficit_lists[idx], start_time_phases)
            phases_list_b = module_b.process_phases(energy_excess_lists[idx], energy_deficit_lists[idx], start_time_phases)

            result_a = extract_results(phases_list_a)
            result_b = extract_results(phases_list_b)

            if not is_equal(result_a, result_b):
                print(f"Versions are not equal at phase count i = {i}")
                return

    print("Versions are equal")

def test_version_solo(version_name, worst_case_scenario = False, phase_count = 5):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)

    print(energy_excess_lists[0]-energy_deficit_lists[0])

    print(energy_excess_lists[0])
    print(energy_deficit_lists[0])

    module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

def visualize_output_each_step(phase_count, version_name = "new_version_fusion_output_each_step", worst_case_scenario = False):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)
    phases_list, mask = module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

    visualize(step_data=phases_list)
