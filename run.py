from helper.hdf5_methodes import compare_simulation_results
from helper.json_methodes import change_cfg, load_config, init_results_folders
from helper.plot_methodes import plot_from_json
from helper.runtime_fitting_methodes import log_log_linear_regression
from helper.test import submethod_analysis, test_versions, test_version_solo
from helper.visualizer import visualize
from main import main

if __name__ == '__main__':

    change_cfg("abort", False)
    cfg = load_config()

    init_results_folders(cfg)



    #main()

    #compare_simulation_results("new_version_fusion", "append_improved_init_capacity_10_numba", cfg)

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg, "new_version_fusion", min_phase_count = 10000, max_phase_count = None)

    #submethod_analysis("append_improved_init_capacity_10_numba")

    #test_versions("append_improved_init_capacity_10_numba_different_datatypes", "append_improved_init_capacity_10_numba")

    #test_version_solo("new_version", True, 200000)

    visualize(cfg,"new_version_fusion" , phase=15, max_cols = 20, case = "average_case")