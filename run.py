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



    #main()

    #compare_simulation_results("new_version_fusion", "append_improved_init_capacity_10_numba", cfg)

    plot_from_json(cfg)

    #log_log_linear_regression(cfg, "new_version_fusion", min_phase_count = 1000, max_phase_count = None)

    #submethod_analysis("new_version_fusion_without_numba")

    #test_versions("new_version_fusion_experimental", "new_version_fusion", phase_count = 1000)

    #test_version_solo("new_version_fusion_experimental", False, 100000)

    #visualize_output_each_step(phase_count =  15)

    #visualize(cfg,"new_version_fusion" , phase=15, max_cols = 100, case = "worst_case")