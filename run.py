from helper.hdf5_methodes import compare_simulation_results
from helper.json_methodes import change_cfg, load_config, init_results_folders
from helper.plot_methodes import plot_from_json
from helper.runtime_fitting_methodes import log_log_linear_regression
from helper.test import submethod_analysis, test_versions, test_version_solo, visualize_output_each_step
from helper.visualizer import visualize
from main import main

if __name__ == '__main__':

    change_cfg("abort", False)
    cfg = load_config()
    init_results_folders(cfg)


    #main(save_to_hdf_till = 10000)

    #compare_simulation_results("new_version_fusion_2", "test_parallel_8_threads", cfg)

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg, "test_parallel_8_threads", min_phase_count = 1000, max_phase_count = None)

    #submethod_analysis("no_numba")

    #test_versions("new_version_fusion_2", "test_parallel_8_threads", start = 1000, end = 1000000, repetitions_count = 1)

    #test_version_solo("new_version_fusion_2", False, 20000000)

    #visualize_output_each_step(phase_count =  13)

    #visualize(cfg,"new_version_fusion_deficit_based" , phase=13, max_cols = 100, case = "average_case")