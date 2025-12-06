import numpy as np


def extract_results(process_phases_output):

    if isinstance(process_phases_output, tuple):
        # return (ids, capacity_excess, capacity_deficit,
        #      size_excess, size_deficit, number_of_excess_not_covered,
        #      starts_excess, starts_deficit,
        #      energy_excess, energy_deficit,
        #      excess_ids)
        starts_excess = process_phases_output[6]
        starts_deficit = process_phases_output[7]
        energy_excess = process_phases_output[8]
        energy_deficit = process_phases_output[9]

        size_excess = process_phases_output[3]
        size_deficit = process_phases_output[4]

        starts_excess_list = [starts_excess[i, :size_excess[i]] for i in range(len(size_excess))]
        starts_deficit_list = [starts_deficit[i, :size_deficit[i]] for i in range(len(size_deficit))]
        energy_excess_list = [energy_excess[i, :size_excess[i]] for i in range(len(size_excess))]
        energy_deficit_list = [energy_deficit[i, :size_deficit[i]] for i in range(len(size_deficit))]

        return starts_excess_list, starts_deficit_list, energy_excess_list, energy_deficit_list

    else:

        expected_attrs = [
            "size_excess", "size_deficit",
            "starts_excess", "starts_deficit",
            "energy_excess", "energy_deficit",
        ]

        starts_excess_list = []
        starts_deficit_list = []
        energy_excess_list = []
        energy_deficit_list = []

        for phase in process_phases_output:

            all_exist = all(hasattr(phase, attr) for attr in expected_attrs)

            if all_exist:

                size_excess = getattr(phase, "size_excess")
                size_deficit = getattr(phase, "size_deficit")
                starts_excess_arr = getattr(phase, "starts_excess")[:size_excess]
                starts_deficit_arr = getattr(phase, "starts_deficit")[:size_deficit]
                energy_excess_arr = getattr(phase, "energy_excess")[:size_excess]
                energy_deficit_arr = getattr(phase, "energy_deficit")[:size_deficit]

            else:

                starts_excess_arr = getattr(phase, "starts_excess")
                starts_deficit_arr = getattr(phase, "starts_deficit")
                energy_excess_arr = getattr(phase, "energy_excess")
                energy_deficit_arr = getattr(phase, "energy_deficit")

            starts_excess_list.append(starts_excess_arr)
            starts_deficit_list.append(starts_deficit_arr)
            energy_excess_list.append(energy_excess_arr)
            energy_deficit_list.append(energy_deficit_arr)

        return starts_excess_list, starts_deficit_list, energy_excess_list, energy_deficit_list