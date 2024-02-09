import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import os
import numpy as np
import pandas as pd

import efes_dataclasses
import effective_energy_shift as efes
import math_energy_systems as mes

class PlotContext:
    def __init__(self, style_context=None, fontsize=None):
        self.fig = None
        self.axs = None

        self.gain_plot_added = False

        self.self_sufficiency_axis_added = False
        self.self_consumption_axis_added = False
        self.axis_self_sufficiency = None
        self.axis_self_consumption = None
        self.position_self_sufficiency_axis = None
        self.position_self_consumption_axis = None
        self.position_additional_y_axis = 1.01

        self.cmap = None
        self.colorbar_added = False
        self.cbar = None

        if style_context is None:
            style_context = ['science', 'ieee']

        self.style_context = style_context

        if fontsize is None:
            fontsize = 16

        self.fontsize = fontsize

    def setup_matplotlib_context(self):
        plt.rcParams['text.usetex'] = True

        plt.rcParams['font.size'] = self.fontsize
        plt.rcParams['legend.fontsize'] = self.fontsize
        plt.rcParams['xtick.labelsize'] = self.fontsize
        plt.rcParams['ytick.labelsize'] = self.fontsize
        plt.rcParams['axes.labelsize'] = self.fontsize

        plt.rcParams['xtick.major.pad'] = 10
        plt.rcParams['ytick.alignment'] = 'center'

    def get_color(self, values):
        return self.cmap(self.cmap_norm(values))

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_frame_on(False)
    ax.axis('off')

def create_plot_frame(figsize=None,
                      height_ratios=None,
                      add_gain_plot=True,
                      grid=True,
                      fontsize=None,
                      ylabel_y=-0.08,
                      no_frame=False
                      ):
    ctx = PlotContext(style_context=['science', 'ieee'], fontsize=fontsize)
    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()

        rows = 1

        if add_gain_plot:
            rows += 1
        if figsize is None:
            figsize = (20, rows * 3)
        else:
            if figsize[0] <= 0:
                figsize = (20, figsize[1])
            if figsize[1] <= 0:
                figsize = (figsize[0], rows * 3)

        fig, axs = plt.subplots(rows, 1, figsize=figsize, sharex=True, height_ratios=height_ratios)

        if rows == 1:
            axs = [axs]

        ctx.fig = fig
        ctx.axs = axs

        row = 0

        axs[row].set_ylabel(ylabel=r'added energy $\mathit{E}^{+}$ [Wh]', x=0, y=0.5, horizontalalignment='center', verticalalignment='top')
        axs[row].yaxis.set_label_coords(ylabel_y, 0.5)
        if grid:
            axs[row].grid()
        if no_frame:
            remove_frame(axs[row])

        if add_gain_plot:
            row += 1
            ctx.gain_plot_added = True
            axs[row].set_ylabel(ylabel=r'gain $\mathit{G}_{\mathrm{day}}$ [1]', x=0, y=0.5, horizontalalignment='center', verticalalignment='top')
            axs[row].yaxis.set_label_coords(ylabel_y, 0.5)
            if grid:
                axs[row].grid()
            if no_frame:
                remove_frame(axs[row])

        axs[-1].set(xlabel=r'capacity $\mathit{C}$ [W]')
    return ctx

def add_self_sufficiency_axis_to_plot(ctx, energy_additional_max, self_sufficiency_initial, energy_demand):

    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()

        ctx.scale_energy, ctx.scale_str_energy = efes.get_scaling(energy_additional_max)

        def func_energy_to_self_sufficiency(energy):
            return efes.mes.calculate_self_sufficiency_from_additional_energy(
                energy_additional=energy,
                energy_demand=energy_demand,
                self_sufficiency_initial=self_sufficiency_initial,
                clip=False
            )

        def func_self_sufficiency_to_energy(self_sufficiency):
            return efes.mes.calculate_additional_energy_from_self_sufficiency(
                self_sufficiency=self_sufficiency,
                energy_demand=energy_demand,
                self_sufficiency_initial=self_sufficiency_initial
            )

        ctx.axis_self_sufficiency = ctx.axs[0].secondary_yaxis(ctx.position_additional_y_axis, functions=(func_energy_to_self_sufficiency, func_self_sufficiency_to_energy))
        ctx.axis_self_sufficiency.set(ylabel=r'self-sufficiency $\psi_{\mathrm{ss}}$ [1]')

    ctx.position_self_sufficiency_axis = ctx.position_additional_y_axis
    ctx.position_additional_y_axis += 0.04 * (float(ctx.fontsize) / 10)
    ctx.self_sufficiency_axis_added = True
    return ctx


def add_self_consumption_axis_to_plot(ctx, energy_additional_max, self_consumption_initial, energy_generation, efficiency_charging, efficiency_discharging):
    import matplotlib.pyplot as plt

    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()
        ctx.scale_energy, ctx.scale_str_energy = efes.get_scaling(energy_additional_max)

        def func_energy_to_self_consumption(energy):
            return efes.mes.calculate_self_consumption_from_additional_energy(
                energy_additional=energy,#/ctx.scale_energy,
                energy_generation=energy_generation,
                self_consumption_initial=self_consumption_initial,
                efficiency_charging=efficiency_charging,
                efficiency_discharging=efficiency_discharging,
                clip=False
            )

        def func_self_consumption_to_energy(self_consumption):
            return efes.mes.calculate_additional_energy_from_self_consumption(
                self_consumption=self_consumption,
                energy_generation=energy_generation,
                self_consumption_initial=self_consumption_initial,
                efficiency_charging=efficiency_charging,
                efficiency_discharging=efficiency_discharging
            )#* ctx.scale_energy

        ctx.axis_self_consumption = ctx.axs[0].secondary_yaxis(ctx.position_additional_y_axis,
                                                           functions=(
                                                               func_energy_to_self_consumption,
                                                               func_self_consumption_to_energy
                                                           ))

        ctx.axis_self_consumption.set(ylabel=r'self-consumption $\psi_{\mathrm{sc}}$ [1]')

    ctx.position_self_consumption_axis = ctx.position_additional_y_axis
    ctx.position_additional_y_axis += 0.04 * (float(ctx.fontsize) / 10)
    ctx.self_consumption_axis_added = True
    return ctx


def setup_colormap(ctx, vmin, vmax, cmap_name=None):
    if cmap_name is None:
        cmap_name = 'jet'
    ctx.cmap_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    ctx.cmap = mpl.cm.get_cmap(cmap_name)
    return ctx


def add_colorbar_to_plot(ctx, label, no_frame=False):
    if ctx.cmap_norm is None:
        raise AttributeError('The colormap has to be initialized by calling setup_colormap(...) first, before the colorbar can be added.')

    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()
        ctx.cbar = ctx.fig.colorbar(mpl.cm.ScalarMappable(norm=ctx.cmap_norm, cmap=ctx.cmap),
                                ax=ctx.axs[0],
                                location='top',
                                anchor=(1.0,0.),
                                shrink=0.5,
                                extend='both'
                                )

        ctx.cbar.ax.set_xlabel(xlabel=label, labelpad=15)
        if no_frame:
            remove_frame(ctx.cbar.ax)

    ctx.colorbar_added = True
    return ctx


def add_scatter_to_axis(ax, x_values, y_values, size=60, marker='o', fillcolor=(1.,1.,1.), edgecolors=(0.,0.,0.), **kwargs):
    ax.scatter(x=x_values, y=y_values, s=size, color=fillcolor, marker=marker, edgecolors=edgecolors, zorder=3, **kwargs)


def add_scatter_at_values(ctx:PlotContext, capacity, energy_additional, cbar_value=None, gain_value=None, **kwargs):
    line_kwargs = dict(
        color=(0.7, 0.7, 0.7),
        linewidth=2,
        linestyle='dotted'
    )

    ctx.axs[0].axvline(x=capacity, zorder=2, **line_kwargs)
    ctx.axs[0].axhline(y=energy_additional, zorder=2, **line_kwargs)
    add_scatter_to_axis(ctx.axs[0], x_values=[capacity], y_values=[energy_additional], **kwargs)

    if ctx.colorbar_added and cbar_value is not None:
        ctx.cbar.ax.axvline(x=cbar_value, zorder=2, **line_kwargs)
        add_scatter_to_axis(ctx.cbar.ax, x_values=[cbar_value], y_values=[0.5], **kwargs)

    if ctx.self_sufficiency_axis_added:
        ctx.axs[0].annotate(' ',
                    xy=(ctx.position_self_sufficiency_axis, energy_additional), xycoords=('axes fraction', 'data'),
                    xytext=(0, 0), textcoords='offset points',
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=3", edgecolor=(0., 0., 0.), facecolor=(1., 1., 1.)),
                    arrowprops=None,
                    annotation_clip=False,
                    color=(1., 1., 1.),
                    fontsize=1
        )

    if ctx.self_consumption_axis_added:
        ctx.axs[0].annotate(' ',
                    xy=(ctx.position_self_consumption_axis, energy_additional), xycoords=('axes fraction', 'data'),
                    xytext=(0, 0), textcoords='offset points',
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=3", edgecolor=(0., 0., 0.), facecolor=(1., 1., 1.)),
                    arrowprops=None,
                    annotation_clip=False,
                    color=(1., 1., 1.),
                    fontsize=1
        )

    if ctx.gain_plot_added:

        ctx.axs[1].axvline(x=capacity, zorder=2, **line_kwargs)
        if gain_value is None:
            gain = float(energy_additional)/capacity
        else:
            gain = gain_value
        ctx.axs[1].axhline(y=gain, zorder=2, **line_kwargs)
        add_scatter_to_axis(ctx.axs[1], x_values=[capacity], y_values=[gain], **kwargs)


def finalize_plot(ctx,
                  title:str = '',
                  axs_settings=None
                  ):

    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()

        ctx.axs[0].legend(frameon=False, bbox_to_anchor=(0.04, 1.0, 1, 0.102), labelspacing=0.5, loc='lower left', ncols=1, borderaxespad=0., mode="expand")
        row = 0

        if ctx.gain_plot_added:
            row += 1
            ctx.axs[row].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., labelspacing=0.5)

        if title != '':
            ctx.fig.suptitle(title, fontsize=1.2*ctx.fontsize)

        if axs_settings is not None:
            def set_ax_settings(ax, ax_settings):
                if ax_settings is None:
                    return
                ax.set(**ax_settings)

            list(map(lambda arg: set_ax_settings(*arg), zip(ctx.axs, axs_settings)))

    return ctx


def save_plot(ctx, filename: str = None, format: str = None, dpi: int = None, **kwargs):
    if format is None:
        format = 'png'
    if dpi is None:
        dpi = 600
    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()
        ctx.fig.savefig(fname=f'{filename}.{format}', dpi=dpi, format=format, **kwargs)


def show_plot(ctx):
    with plt.style.context(ctx.style_context):
        ctx.setup_matplotlib_context()
        plt.show()


def create_simple_result_plot(results: efes.efes_dataclasses.Results, figsize=None, axs_settings=None,
                              add_self_sufficiency_axis=True, add_self_consumption_axis=True,
                              add_gain_plot=True,
                              title:str = '',
                              height_ratios=None,
                              fontsize=None,
                              linewidth=None,
                              ylabel_y=-0.1,
                              grid=True,
                              no_frame=False,
                              filename: str = None,
                              format: str = None,
                              dpi: int = None,
                              show=True):

    ctx = create_plot_frame(figsize=figsize,
                            height_ratios=height_ratios,
                            add_gain_plot=add_gain_plot,
                            fontsize=fontsize,
                            ylabel_y=ylabel_y,
                            grid=grid,
                            no_frame=no_frame
                            )

    capacity = results.analysis_results.capacity if results.query_results is None else results.query_results[0].capacity
    energy_additional = results.analysis_results.energy_additional if results.query_results is None else results.query_results[0].energy_additional
    energy_demand = results.analysis_results.energy_demand
    energy_generation = results.analysis_results.energy_generation

    capacity_max = results.analysis_results.capacity_max
    energy_additional_max = results.analysis_results.energy_additional_max
    self_sufficiency_max = results.analysis_results.self_sufficiency_max
    self_consumption_max = results.analysis_results.self_consumption_max

    scale_energy, scale_str_energy = efes.get_scaling(energy_additional_max)
    scale_capacity, _ = efes.get_scaling(capacity[-1])
    self_sufficiency_initial = results.analysis_results.self_sufficiency_initial
    self_consumption_initial = results.analysis_results.self_consumption_initial

    efficiency_charging = results.analysis_results.data_input.efficiency_charging
    efficiency_discharging = results.analysis_results.data_input.efficiency_charging

    row = 0
    ctx.axs[row].hlines(xmin=0, xmax=capacity[-1], y=0, linestyles='-.', color='black', linewidth=linewidth, label=r'$\psi_{\mathrm{ss,0}}=' + f'{self_sufficiency_initial:.2f}$, ' + r'$\psi_{\mathrm{sc,0}}=' + f'{self_consumption_initial:.2f}$')
    ctx.axs[row].plot(capacity, energy_additional, linewidth=linewidth, color='black', label=r'$E^{+}$')#,r'$\energyAdditional$')

    label_max = r'$\mathit{E}^{+}_{\mathrm{mmax}}(\mathit{C}=' + efes.pretty_print(capacity_max, "Wh") + ')=' + f'{scale_energy * energy_additional_max:.2f}$' + scale_str_energy + 'Wh' + r' $\longrightarrow$ ' + r'$\psi_{\mathrm{ss,max}}=' + f'{self_sufficiency_max:.2f}$, ' + r'$\psi_{\mathrm{sc,max}}=' + f'{self_consumption_max:.2f}$'
    ctx.axs[row].hlines(xmin=0, xmax=capacity[-1], y=energy_additional_max, linestyles='--', color='black', linewidth=linewidth, label=label_max)

    if add_self_sufficiency_axis:
        ctx = add_self_sufficiency_axis_to_plot(ctx,
                                                energy_additional_max=energy_additional_max,
                                                self_sufficiency_initial=self_sufficiency_initial,
                                                energy_demand=energy_demand
                                                )
    if add_self_consumption_axis:
        ctx = add_self_consumption_axis_to_plot(ctx,
                                                energy_additional_max=energy_additional_max,
                                                self_consumption_initial=self_consumption_initial,
                                                energy_generation=energy_generation,
                                                efficiency_charging=efficiency_charging,
                                                efficiency_discharging=efficiency_discharging
                                                )

    if add_gain_plot:
        row += 1
        gain_per_day = results.query_results[0].gain_per_day
        ctx.axs[row].plot(capacity, gain_per_day, linewidth=linewidth, color='black', label=r'$\\mathir{G}_{\mathrm{day}}$')

    finalize_plot(ctx,
                  title=title,
                  axs_settings=axs_settings
                  )

    if filename != '':
        save_plot(ctx, filename=filename, format=format, dpi=dpi)

    if show:
        show_plot(ctx)

    return ctx


def create_variation_plot(parameter_study_results: efes_dataclasses.ParameterStudyResults,
                          cbar_label: str,
                          cmap_parameter_name: str=None,
                          cmap_name:str = 'jet',
                          index_reference_result: int = -1,
                          figsize=None,
                          add_self_sufficiency_axis=True,
                          add_self_consumption_axis=True,
                          add_gain_plot=True,
                          add_line_for_initial_values=True,
                          add_line_for_maximum_values=True,
                          height_ratios=None,
                          fontsize:float = None,
                          linewidth:float = 2,
                          ylabel_y:float = -0.06,
                          use_fill:bool = True,
                          grid=True,
                          no_frame=False,
                          filename:str = None,
                          dpi:int = None,
                          format:str = None,
                          show:bool = True,
                          axs_settings=None,
                          title:str = ''
                          ):

    if cmap_parameter_name is None:
        cmap_parameter_name = parameter_study_results.parameter_variation.columns[0]

    ctx = create_plot_frame(figsize=figsize,
                            height_ratios=height_ratios,
                            add_gain_plot=add_gain_plot,
                            fontsize=fontsize,
                            ylabel_y=ylabel_y,
                            grid=grid,
                            no_frame=no_frame
                            )

    reference_result = parameter_study_results.results[index_reference_result]
    reference_result_needs_loading = isinstance(reference_result, str)
    if reference_result_needs_loading:
        reference_result = efes_dataclasses.unpickle(reference_result)

    capacity_max = reference_result.analysis_results.capacity_max
    energy_additional_max = reference_result.analysis_results.energy_additional_max
    self_sufficiency_max = reference_result.analysis_results.self_sufficiency_max
    self_consumption_max = reference_result.analysis_results.self_consumption_max
    energy_demand = reference_result.analysis_results.energy_demand
    energy_generation = reference_result.analysis_results.energy_generation

    scale_energy, scale_str_energy = efes.get_scaling(energy_additional_max)

    self_sufficiency_initial = reference_result.analysis_results.self_sufficiency_initial
    self_consumption_initial = reference_result.analysis_results.self_consumption_initial

    efficiency_charging = reference_result.analysis_results.data_input.efficiency_charging
    efficiency_discharging = reference_result.analysis_results.data_input.efficiency_charging

    if add_self_sufficiency_axis:
        ctx = add_self_sufficiency_axis_to_plot(ctx,
                                                energy_additional_max=energy_additional_max,
                                                self_sufficiency_initial=self_sufficiency_initial,
                                                energy_demand=energy_demand
                                                )
    if add_self_consumption_axis:
        ctx = add_self_consumption_axis_to_plot(ctx,
                                                energy_additional_max=energy_additional_max,
                                                self_consumption_initial=self_consumption_initial,
                                                energy_generation=energy_generation,
                                                efficiency_charging=efficiency_charging,
                                                efficiency_discharging=efficiency_discharging
                                                )

    with pd.option_context('mode.use_inf_as_na', True):
        print(parameter_study_results)
        vmin = parameter_study_results.parameter_variation[cmap_parameter_name].min()
        vmax = parameter_study_results.parameter_variation[cmap_parameter_name].max()

        ctx = setup_colormap(ctx,
                             vmin=vmin,
                             vmax=vmax,
                             cmap_name=cmap_name)

    ctx = add_colorbar_to_plot(ctx, label=cbar_label, no_frame=no_frame)
    if add_line_for_initial_values:
        ctx.axs[0].hlines(xmin=0, xmax=reference_result.query_results[0].capacity[-1], y=0, linestyles='-.', color='black',
                          linewidth=linewidth,
                          label=r'$\psi_{\mathrm{ss,0}}=' + f'{self_sufficiency_initial:.2f}$, ' + r'$\psi_{\mathrm{sc,0}}=' + f'{self_consumption_initial:.2f}$'
                          )

    ctx.axs[0].plot(reference_result.query_results[0].capacity, reference_result.query_results[0].energy_additional,
                    linestyle='-', linewidth=1.5*linewidth, color='black',
                    label=r'$\mathit{E}^{+}(\mathit{C})$',
                    zorder=2)

    label_max = r'$\mathit{E}^{+}(\mathit{C} = \text{' + efes.pretty_print(capacity_max, "Wh") + '})=' + f'{scale_energy * energy_additional_max:.2f}$' + scale_str_energy + 'Wh' + r' $\longrightarrow$ ' + r'$\psi_{\mathrm{ss,max}}=' + f'{self_sufficiency_max:.2f}$, ' + r'$\psi_{\mathrm{sc,max}}=' + f'{self_consumption_max:.2f}$'
    if add_line_for_maximum_values:
        ctx.axs[0].hlines(xmin=0, xmax=reference_result.query_results[0].capacity[-1],
                          y=energy_additional_max, linestyles='--', color='black',
                          linewidth=linewidth,
                          label=label_max
                          )

    scale_capacity, _ = efes.get_scaling(reference_result.query_results[0].capacity[-1])

    for variation, results in zip(parameter_study_results.parameter_variation.to_dict(orient='records')[::-1],
                                  parameter_study_results.results[::-1]):
        results_needs_loading = isinstance(results, str)
        if results_needs_loading:
            results = efes_dataclasses.unpickle(results)

        color = ctx.get_color(variation[cmap_parameter_name])

        row = 0
        capacity = results.query_results[0].capacity
        energy_additional = results.query_results[0].energy_additional

        if use_fill:
            ctx.axs[0].fill_between(capacity, energy_additional, y2=0, color=color)
        else:
            ctx.axs[0].plot(capacity, energy_additional, linestyle='-', linewidth=linewidth, color=color)

        if add_gain_plot:
            row += 1

            gain_per_day = results.query_results[0].gain_per_day

            if use_fill:
                ctx.axs[row].fill_between(capacity, gain_per_day, y2=0, color=color)
            else:
                ctx.axs[row].plot(capacity, gain_per_day, linestyle='-', linewidth=linewidth, color=color)

        if results_needs_loading:
            del results

    if reference_result_needs_loading:
        del reference_result

    ctx = finalize_plot(ctx,
                  title=title,
                  axs_settings=axs_settings
                  )

    if filename is not None and filename != '':
        save_plot(ctx, filename=filename, format=format, dpi=dpi)

    if show:
        show_plot(ctx)

    return ctx
