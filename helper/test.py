import cProfile
import pstats
from helper.compare_methodes import test_results
from helper.visualizer import visualize
from main import init, import_version

delim = "-"*100


def submethod_analysis(version_name, worst_case_scenario = False, phase_count = 40000):

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

def test_versions(version_name_a, version_name_b, worst_case_scenario = False, phase_count = 1000):

    module_a = import_version(version_name_a)
    module_b = import_version(version_name_b)

    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)

    phases_list_a = module_a.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)
    phases_list_b = module_b.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

    test_results([phases_list_a], [phases_list_b])

def test_version_solo(version_name, worst_case_scenario = False, phase_count = 5):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)
    phases_list = module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

def visualize_output_each_step(version_name = "new_version_fusion_output_each_step", worst_case_scenario = False, phase_count = 53):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)
    phases_list = module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

    visualize(step_data=phases_list)
