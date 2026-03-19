from helper.json_methodes import change_cfg, load_config, init_results_folders
from helper.plot_methodes import plot_from_json
from helper.runtime_fitting_methodes import log_log_linear_regression
from helper.test import submethod_analysis, test_versions, test_version_solo, visualize_output_each_step
from helper.visualizer import visualize
from main import main
from helper.csv import convert_jsons_to_csv

if __name__ == '__main__':

    change_cfg("abort", False)
    cfg = load_config()
    init_results_folders(cfg)

    main(save_to_hdf_till = 10000)

    #to_plot = ["new_version_fusion_phaseless", "new_version_fusion"]
    #plot_from_json(cfg, to_plot)

    #convert_jsons_to_csv()

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg, "aa_mefes_2", min_phase_count = 10)

    #submethod_analysis("aa_mefes", phase_count = 10000)

    #blacklist = ["deficit_based"]
    #to_test = [v for v in cfg['versions'] if v not in blacklist]
    #test_versions(to_test, start =  10, end =  1000, repetitions_count = 5, worst_case_scenario = False)

    #test_version_solo("aa_compress", False, 95)

    #visualize_output_each_step(phase_count =  13)

    #visualize(cfg,"original" , phase=10, max_cols = 100, case = "average_case")