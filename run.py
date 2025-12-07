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

    #compare_simulation_results("new_version_fusion_classless_parallel", "new_version_fusion_classless_parallel_with_flags", cfg)

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg, "new_version_fusion_classless_parallel_with_flags", min_phase_count = 1000, max_phase_count = None)

    submethod_analysis("new_version_fusion_classless_parallel_with_flags_no_numba")

    #test_versions("original", "new_version_fusion_classless_parallel_with_flags")

    #test_version_solo("new_version_fusion", False, 271)

    #visualize_output_each_step(phase_count =  271)

    #visualize(cfg,"new_version_fusion_classless_parallel_with_flags" , phase=271, max_cols = 10, case = "average_case")