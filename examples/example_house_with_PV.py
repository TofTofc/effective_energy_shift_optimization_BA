import pandas as pd
import numpy as np

import effective_energy_shift as efes
import efes_plotting as efes_plot

def read_example_data():
    df = pd.read_csv('example_house_with_PV_3_years.csv', sep=';')
    df['time'] = pd.to_timedelta(df['time'])
    time = df['time']
    time_in_seconds = df['time'].dt.total_seconds().to_numpy() / 3600
    delta_time_step = time_in_seconds[1] - time_in_seconds[0]
    power_demand = df['dem'].to_numpy()
    power_generation = df['gen'].to_numpy()
    return dict(time=time, time_in_seconds=time_in_seconds, power_generation=power_generation, power_demand=power_demand, delta_time_step=delta_time_step)
def run_power_variation():
    input_data = read_example_data()
    efficiency_discharging = 0.95
    efficiency_charging = 0.95
    efficiency_direct_usage = 0.95
    capacity_target = np.linspace(0.01,20000,200)

    filename = 'house_example_results'

    parameter_study_results = efes.run_parameter_study(
        power_generation=input_data['power_generation'],
        power_demand=input_data['power_demand'],
        delta_time_step=input_data['delta_time_step'],
        efficiency_direct_usage=efficiency_direct_usage,
        efficiency_discharging=efficiency_discharging,
        efficiency_charging=efficiency_charging,
        capacity_target=capacity_target,
        parameter_variation=pd.DataFrame(data=dict(
            power_max_discharging=np.array([*np.linspace(200, 1000, 5), np.inf]),
            power_max_charging=np.array([*np.linspace(200, 1000, 5), np.inf]),
        )),
        result_dir=filename
    )

    print(f'Results of parameter study saved in subfolder {filename}')
    print(f'Creating plots...')

    def finalize_plots(fig, axs):
        fig.savefig(fname=f'{filename}/{filename}.png', dpi=600, format='png', transparent=True)
        print(f'Plots saved in subfolder {filename}/{filename}.png')
        return fig, axs

    ctx = efes_plot.create_variation_plot(parameter_study_results,
                                       cmap_parameter_name='power_max_charging',
                                       cbar_label='$\mathit{P}_{\mathrm{ch,max}}$ and $\mathit{P}_{\mathrm{dch,max}}$ [W]',
                                       xlim=[0,15000],
                                       ylims=[None, [0,1500], [0,1500]],
                                       final_callback=finalize_plots
                                       )



if __name__ == '__main__':
    run_power_variation()
