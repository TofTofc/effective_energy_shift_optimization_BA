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

    #compare_simulation_results("new_version_fusion_phaseless_2", "new_version_fusion_phaseless_2_parallel_no_resize", cfg)

    #to_plot = ["aa_compress", "new_version_fusion_phaseless_2_array_structure"]
    #plot_from_json(cfg, to_plot)

    #log_log_linear_regression(cfg, "aa_compress", min_phase_count = 5000)

    #submethod_analysis("new_version_fusion_phaseless_2_array_structure")

    test_versions("original", "new_version_fusion_phaseless_2_array_structure", start = 25, end = 500, repetitions_count = 1, worst_case_scenario = False)

    #test_version_solo("aa_compress", False, 95)

    #visualize_output_each_step(phase_count =  13)

    #visualize(cfg,"new_version_fusion_phaseless_2_array_structure" , phase=10, max_cols = 100, case = "average_case")