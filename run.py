from helper.hdf5_methodes import compare_simulation_results
from helper.json_methodes import change_cfg, load_config, init_results_folders
from helper.plot_methodes import plot_from_json
from helper.runtime_fitting_methodes import log_log_linear_regression
from helper.test import submethod_analysis, test_versions
from main import main

if __name__ == '__main__':

    change_cfg("abort", False)
    cfg = load_config()

    init_results_folders(cfg)



    #main()

    #compare_simulation_results("original_simplified", "append_improved_init_capacity_10_numba", cfg)

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg, "append_improved_init_capacity_10_numba")

    #submethod_analysis("append_improved_init_capacity_10_numba")

    #test_versions("original", "append_improved_init_capacity_10_numba")