import cProfile
import pstats
from helper.compare_methodes import is_equal
from helper.extract_results import extract_results
from helper.visualizer import visualize
from main import init, import_version

delim = "-"*100


def submethod_analysis(version_name, worst_case_scenario = False, phase_count = 100000):

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


def test_versions(versions_to_test, start=10, end=1000, repetitions_count=1, worst_case_scenario=False):

    original_module = import_version("original")
    test_modules = {name: import_version(name) for name in versions_to_test}
    failed_versions = set()

    for i in range(start, end):

        print(f"Testing Phase count: {i}")

        energy_excess_lists, energy_deficit_lists, start_time_phases = init(
            worst_case_scenario, 125, i, repetitions_count
        )

        for idx in range(repetitions_count):

            phases_orig = original_module.process_phases(
                energy_excess_lists[idx], energy_deficit_lists[idx], start_time_phases
            )
            result_orig = extract_results(phases_orig)

            for version_name in versions_to_test:
                current_module = test_modules[version_name]

                phases_test = current_module.process_phases(
                    energy_excess_lists[idx], energy_deficit_lists[idx], start_time_phases
                )
                result_test = extract_results(phases_test)

                if not is_equal(result_orig, result_test):
                    print(f"Version '{version_name}' failed at phase count {i}")
                    failed_versions.add(version_name)

    print(delim)
    for name in versions_to_test:
        status = "FAILED" if name in failed_versions else "PASSED"
        print(f"Version {name}: {status}")

def test_version_solo(version_name, worst_case_scenario = False, phase_count = 5):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)

    module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

def visualize_output_each_step(phase_count, version_name = "x_new_version_fusion_output_each_step", worst_case_scenario = False):

    module = import_version(version_name)
    energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, 125, phase_count, 1)
    phases_list, mask = module.process_phases(energy_excess_lists[0], energy_deficit_lists[0], start_time_phases)

    visualize(step_data=phases_list)
