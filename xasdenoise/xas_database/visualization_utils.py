"""
Multi-spectrum visualization functions.

This module contains plotting functions that operate on lists of Spectrum objects.
These functions use the single spectrum visualization functions as building blocks
and add multi-spectrum specific features like vertical displacement.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union
from xasdenoise.xas_data.spectrum import Spectrum
from xasdenoise.xas_data.visualization import (
    setup_plot, legend_exists, descriptor_plot, 
    edge_plot, get_crop, spectrum_plot
)

def finalize_plot(yax_lim: Optional[tuple] = None, 
                  displace_vertically: bool = False,
                  displace_time_vertically: bool = False,
                  vertical_displacement_offset: float = 0.5) -> None:
    """
    Finalize the plot by adding a legend and adjusting the layout.

    Args:
        yax_lim (Optional[tuple]): Limits for the y-axis. Defaults to None.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
    """
    if displace_vertically:
        fig = plt.gcf()
        displace_spectra_vertically(fig, vertical_displacement_offset)
    elif displace_time_vertically:
        fig = plt.gcf()
        displace_time_instances_vertically(fig, vertical_displacement_offset)
    else:
        if legend_exists():
            plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if yax_lim is not None:
            plt.ylim(yax_lim)
        plt.show()

def displace_spectra_vertically(fig: plt.Figure, offset_increment: float = 0.3) -> None:
    """
    Displace spectra vertically in an existing figure and add labels at the start of each spectrum.

    Args:
        fig (plt.Figure): The existing figure containing spectra.
        offset_increment (float): Increment for vertical displacement. Defaults to 0.3.
    """
    from matplotlib.collections import LineCollection

    ax = fig.axes[0]
    lines = ax.get_lines()

    offset = 0

    # adjust the lines upwards
    for line in lines:
        try:
            y_data = line.get_ydata()
            line.set_ydata(y_data + offset)
            ax.text(line.get_xdata()[0], y_data[0] + offset, line.get_label(), fontsize='small', verticalalignment='bottom')
            offset += offset_increment
        except:
            continue
                
    # Adjust figure size based on the number of spectra
    num_spectra = len(lines)
    current_size = fig.get_size_inches()
    new_height = current_size[1] + num_spectra * offset_increment
    fig.set_size_inches(current_size[0], new_height)
    
    ax = fig.axes[0]
    ylim = list(ax.get_ylim())
    ylim[1] += num_spectra * offset_increment
    
    # adjust edge and descriptor vlines
    vlines_labels = []
    vlines = [child for child in ax.get_children() if isinstance(child, LineCollection) and hasattr(child, '_is_descriptor_line')]
    for vline in vlines:
        try:
            segments = vline.get_segments()
            new_segments = []
            for segment in segments:
                x,y = segment
                y[1] += num_spectra * offset_increment
                new_segment = [x,y]
                new_segments.append(new_segment)
            vline.set_segments(new_segments)
        
            # Add label at the top of the vline
            label = vline.get_label()
            if label: ax.text(x[0], y[1], label, fontsize='small', verticalalignment='top')
        except:
            continue
        
    plt.ylim(ylim)
    plt.tight_layout()
    # plt.gca().xaxis.grid(True)
    plt.show()


def displace_time_instances_vertically(fig: plt.Figure, offset_increment: float = 0.5) -> None:
    """
    Displace time instances vertically in an existing figure and add labels at the start of each time instance.

    Args:
        fig (plt.Figure): The existing figure containing spectra.
        offset_increment (float): Increment for vertical displacement. Defaults to 0.05.
    """
    ax = fig.axes[0]
    lines = ax.get_lines()

    offset = 0
    # adjust the lines upwards
    for t_idx, line in enumerate(lines):
        try:
            y_data = line.get_ydata()
            line.set_ydata(y_data + offset)
            label = f'time {t_idx}'
            ax.text(line.get_xdata()[0], y_data[0] + offset, label, fontsize='small', verticalalignment='bottom')
            offset += offset_increment
            
        except:
            continue
    height = y_data.max() + offset            

    
    # Adjust figure size based on the number of time instances
    # num_instances = len(lines)
    # current_size = fig.get_size_inches()
    # new_height = height
    # fig.set_size_inches(current_size[0], new_height)
    
    ax = fig.axes[0]
    ylim = list(ax.get_ylim())
    ylim[1] = height
    
    plt.yticks([])
    plt.ylabel('Spectra')
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def get_yax_lim(spectrum_list: List[Spectrum]) -> List[float]:
    """
    Get the y-axis limits for a list of spectra.

    Args:
        spectrum_list (List[Spectrum]): List of Spectrum objects.

    Returns:
        List[float]: Minimum and maximum y-axis limits.
    """
    ymin = np.min([np.min(data.spectrum) for data in spectrum_list])
    ymax = np.max([np.max(data.spectrum) for data in spectrum_list])
    return [ymin, ymax]


"""
Multi-spectrum plotting functions
"""

def plot_spectrum(
    data: Union[Spectrum, List[Spectrum]],
    title: str = '',
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
    time_index: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
    remove_labels: bool = False,
    center_on_edge: bool = False,
) -> None:
    """
    Plot the spectrum for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to ''.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace multiple spectra above each other. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
        remove_labels (bool): Whether to remove labels. Defaults to False.
        center_on_edge (bool): Whether to center the plot on the edge. Defaults to False.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        label = None if remove_labels else getattr(d, 'compound', '__nolegend__')
        crop = get_crop(d.energy, d.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None

        if center_on_edge:
            d = d.copy()
            d.energy -= d.edge

        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, crop, time_index)
        else:
            spectrum_plot(d.energy, d.spectrum, label, crop, time_index)

    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_edge(
    data: Union[Spectrum, List[Spectrum]],
    title: str = 'Zoom to spectrum edge',
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
    time_index: Optional[int] = None,
    crop_min: int = 50,
    crop_max: int = 50,
) -> None:
    """
    Plot the spectrum edge for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to 'Zoom to spectrum edge'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (int): Minimum crop around the edge. Defaults to 50.
        crop_max (int): Maximum crop around the edge. Defaults to 50.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        crop = get_crop(d.energy, d.edge, crop_min, crop_max)
        label = getattr(d, 'compound', '__nolegend__')
        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, crop, time_index)
        else:
            spectrum_plot(d.energy, d.spectrum, label, crop, time_index)
    
    # Use the last spectrum's edge for the edge plot
    edge_plot(data[-1].edge)
    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_background(
    data: Union[Spectrum, List[Spectrum]],
    title: str = 'Spectrum background',
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
    time_index: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
) -> None:
    """
    Plot the background region for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to 'Background region'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        crop = get_crop(d.energy, d.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None
        label = getattr(d, 'compound', '__nolegend__')
        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, crop, time_index)
        else:
            spectrum_plot(d.energy, d.spectrum, label, crop, time_index)
            
    # Plot background for the last spectrum
    label = 'background'
    spectrum_plot(data[-1].energy, data[-1].background, label, crop, time_index)
    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_xanes(
    data: Union[Spectrum, List[Spectrum]],
    title: str = 'XANES region',
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
) -> None:
    """
    Plot the XANES region for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to 'XANES region'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        label = getattr(d, 'compound', '__nolegend__')
        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, d.xanes_region_indices)
        else:
            spectrum_plot(d.energy, d.spectrum, label, d.xanes_region_indices)
    
    edge_plot(data[-1].edge)
    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_exafs(
    data: Union[Spectrum, List[Spectrum]],
    title: str = 'EXAFS region',
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
) -> None:
    """
    Plot the EXAFS region for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to 'EXAFS region'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        label = getattr(d, 'compound', '__nolegend__')
        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, d.exafs_region_indices)
        else:
            spectrum_plot(d.energy, d.spectrum, label, d.exafs_region_indices)
    
    edge_plot(data[-1].edge)
    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_descriptor(
    data: Union[Spectrum, List[Spectrum]],
    descriptor: str,
    title: str = 'Zoom to spectrum edge',
    crop_min: int = 50,
    crop_max: int = 50,
    time_averaged: bool = True,
    displace_vertically: bool = False,
    displace_time_vertically: bool = False,
    vertical_displacement_offset: float = 0.5,
) -> None:
    """
    Plot the spectrum edge with a descriptor for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        descriptor (str): Descriptor to highlight on the plot.
        title (str): Title of the plot. Defaults to 'Zoom to spectrum edge'.
        crop_min (int): Minimum crop around the edge. Defaults to 50.
        crop_max (int): Maximum crop around the edge. Defaults to 50.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
    """
    setup_plot(title)
    if not isinstance(data, list):
        data = [data]

    for d in data:
        crop = get_crop(d.energy, d.edge, crop_min, crop_max)
        label = getattr(d, 'compound', '__nolegend__')
        if time_averaged and not displace_time_vertically:
            spectrum_plot(d.energy, d.time_averaged_spectrum, label, crop)
        else:
            spectrum_plot(d.energy, d.spectrum, label, crop)

    colors = [line.get_color() for line in plt.gca().lines]
    for idx, d in enumerate(data):
        descriptor_plot(getattr(d, descriptor), descriptor, colors[idx])

    edge_plot(data[-1].edge)
    finalize_plot(displace_vertically=displace_vertically, displace_time_vertically=displace_time_vertically, vertical_displacement_offset=vertical_displacement_offset)


def plot_spectrum_time_instances(
    data: Union[Spectrum, List[Spectrum]],
    title: str = '',
    vertical_displacement_offset: float = 0.5,
    time_instance_number: Optional[int] = None,
    time_binning_size: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
) -> None:
    """
    Plot the spectrum for a single Spectrum object or a list of Spectrum objects.

    Args:
        data (Union[Spectrum, List[Spectrum]]): The spectrum data.
        title (str): Title of the plot. Defaults to ''.
        vertical_displacement_offset (float): Offset for vertical displacement. Defaults to 0.5.
        time_instance_number (Optional[int]): Which time instance to plot. Defaults to None.
        time_binning_size (Optional[int]): Size of the time binning. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
    """
    setup_plot(title)

    # For time instances, we typically work with a single spectrum
    if isinstance(data, list):
        data = data[0]  # Use the first spectrum

    label = getattr(data, 'compound', '__nolegend__')
    crop = get_crop(data.energy, data.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None

    if time_instance_number is not None:
        time_indices = np.linspace(0, data.spectrum.shape[1]-1, time_instance_number).astype(int)
    else: 
        time_indices = None
        
    if time_binning_size is not None:
        data = data.copy()
        data.bin_time_instances(time_binning_size)
                
    spectrum_plot(data.energy, data.spectrum, label, crop, time_indices)
    finalize_plot(displace_time_vertically=True, vertical_displacement_offset=vertical_displacement_offset)
