import pandas as pd
import numpy as np
import time

import effective_energy_shift as efes

def read_example_data():
    df = pd.read_csv('example_house_with_PV_3_years.csv', sep=';')
    df['time'] = pd.to_timedelta(df['time'])
    time = df['time']
    time_in_seconds = df['time'].dt.total_seconds().to_numpy() / 3600
    delta_time_step = time_in_seconds[1] - time_in_seconds[0]
    power_demand = df['dem'].to_numpy()
    power_generation = df['gen'].to_numpy()
    return dict(time=time, time_in_seconds=time_in_seconds, power_generation=power_generation, power_demand=power_demand, delta_time_step=delta_time_step)

if __name__ == '__main__':
    input_data = read_example_data()
    efficiency_discharging = 0.95
    efficiency_charging = 0.95
    efficiency_direct_usage = 0.95
    #capacity_target = np.linspace(0.01, 20000, 100)
    power_max = np.inf

    t0 = time.time()
    results = efes.perform_energy_storage_dimensioning(
        power_max_charging=power_max,
        power_max_discharging=power_max,
        power_generation=input_data['power_generation'],
        power_demand=input_data['power_demand'],
        delta_time_step=input_data['delta_time_step'],
        efficiency_direct_usage=efficiency_direct_usage,
        efficiency_discharging=efficiency_discharging,
        efficiency_charging=efficiency_charging,
        #capacity_target=capacity_target
    )
    print(time.time() - t0)
