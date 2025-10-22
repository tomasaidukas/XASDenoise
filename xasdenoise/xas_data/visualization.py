"""
Single spectrum visualization functions.

This module contains plotting functions that operate on individual Spectrum objects.
These functions are the base building blocks that will be used by the multi-spectrum
visualization functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from xasdenoise.xas_data.spectrum import Spectrum


def setup_plot(title: str, figsize: tuple = (8, 6), xlabel: str = 'Energy (eV)', ylabel: str = 'Absorption') -> None:
    """
    Set up the plot with a title and axis labels.

    Args:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis. Defaults to 'Energy (eV)'.
        ylabel (str): Label for the y-axis. Defaults to 'Absorption'.
    """
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def legend_exists():
    """Check if there are any artists with labels that can form a legend"""
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    # Filter out empty labels and underscore labels (matplotlib convention)
    valid_labels = [label for label in labels if label and not label.startswith('_')]
    return len(handles) > 0 and len(valid_labels) > 0

def finalize_plot(yax_lim: Optional[tuple] = None, 
                  displace_time_vertically: bool = False,
                  vertical_displacement_offset: float = 0.5) -> None:
    """
    Finalize a single spectrum plot by adding a legend and adjusting the layout.

    Args:
        yax_lim (Optional[tuple]): Limits for the y-axis. Defaults to None.
        displace_vertically (bool): Whether to displace spectra vertically. Defaults to False.
        displace_time_vertically (bool): Whether to displace time instances vertically. Defaults to False.
    """
    if displace_time_vertically:
        fig = plt.gcf()
        displace_time_instances_vertically(fig, vertical_displacement_offset)
    else:
        if legend_exists():
            plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if yax_lim is not None:
            plt.ylim(yax_lim)
        plt.show()


def displace_time_instances_vertically(fig: plt.Figure, offset_increment: float = 0.5, 
                                       labels: Optional[str] = None) -> None:
    """
    Displace time instances vertically in an existing figure and add labels at the start of each time instance.

    Args:
        fig (plt.Figure): The existing figure containing spectra.
        offset_increment (float): Increment for vertical displacement. Defaults to 0.05.
    """
    ax = fig.axes[0]
    lines = ax.get_lines()
    if labels is None:
        labels = [f'time {t_idx}' for t_idx in range(len(lines))]
        
    offset = 0
    # adjust the lines upwards
    for t_idx, line in enumerate(lines):
        try:
            y_data = line.get_ydata()
            line.set_ydata(y_data + offset)
            label = labels[t_idx]
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

def spectrum_plot(
    energy: np.ndarray,
    spectrum: np.ndarray,
    label: str,
    crop: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
) -> None:
    """
    Helper function to plot spectrum data with appropriate labels.

    Args:
        energy (np.ndarray): Array of energy values.
        spectrum (np.ndarray): Array of spectrum values.
        label (str): Label for the plot.
        crop (Optional[np.ndarray]): Indices for cropping the data. Defaults to None.
        time_index (Optional[int]): Specific time index to plot. Defaults to None.
    """
    if crop is not None:
        energy = energy[crop]
        spectrum = spectrum[crop, :] if spectrum.ndim > 1 else spectrum[crop]

    if time_index is not None:
        spectrum = spectrum[:, time_index]

    # For time-series, use one label and set others to None
    labels = label if spectrum.ndim == 1 else [label] + [None] * (spectrum.shape[1] - 1)

    plt.plot(energy, spectrum, label=labels)


def spectrum_plot_time_gradient(
    energy: np.ndarray,
    spectrum: np.ndarray,
    label: str,
    crop: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    cmap: str = 'plasma',
) -> None:
    """
    Helper function to plot spectrum data with appropriate labels.

    Args:
        energy (np.ndarray): Array of energy values.
        spectrum (np.ndarray): Array of spectrum values.
        label (str): Label for the plot.
        crop (Optional[np.ndarray]): Indices for cropping the data. Defaults to None.
        time_index (Optional[int]): Specific time index to plot. Defaults to None.
        time_gradient (bool): Whether to use gradient colors for time instances. Defaults to False.
        cmap (str): Colormap for gradient plotting. Defaults to 'plasma'.
    """
    if crop is not None:
        energy = energy[crop]
        spectrum = spectrum[crop, :] if spectrum.ndim > 1 else spectrum[crop]

    if time_index is not None:
        spectrum = spectrum[:, time_index]

    # If gradient is requested and we have time-series data, plot with colors from colormap
    if spectrum.ndim > 1:
        cmap_obj = plt.get_cmap(cmap)
        n_times = spectrum.shape[1]
        colors = [cmap_obj(i / max(1, n_times - 1)) for i in range(n_times)]
        
        # Plot each time instance as its own line with gradient color
        for idx in range(n_times):
            y = spectrum[:, idx]
            line_label = label if idx == 0 else None  # Only label the first line
            plt.plot(energy, y, color=colors[idx], label=line_label, alpha=0.5)
        
        # Add colorbar with proper axes reference
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=0, vmax=n_times-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Instance')
        # Set colorbar ticks to show actual time indices
        if n_times <= 10:
            cbar.set_ticks(range(n_times))
            cbar.set_ticklabels([str(i) for i in range(n_times)])
        else:
            # For many time instances, show fewer ticks
            tick_indices = np.linspace(0, n_times-1, 5).astype(int)
            cbar.set_ticks(tick_indices)
            cbar.set_ticklabels([str(i) for i in tick_indices])
    else:
        # Standard plotting behavior
        labels = label if spectrum.ndim == 1 else [label] + [None] * (spectrum.shape[1] - 1)
        plt.plot(energy, spectrum, label=labels)

def descriptor_plot(val: float, descriptor: str, colors: str) -> None:
    """
    Plot a vertical line for a descriptor.

    Args:
        val (float): Value of the descriptor.
        descriptor (str): Name of the descriptor.
        colors (str): Color of the line.
    """
    ylim = plt.gca().get_ylim()
    line = plt.vlines(val, ylim[0], ylim[1], colors, 'dashed', label=f'{descriptor} = {val} eV')
    line._is_descriptor_line = True


def edge_plot(edge: float) -> None:
    """
    Plot a vertical line for the edge.

    Args:
        edge (float): Energy value of the edge.
    """
    ylim = plt.gca().get_ylim()
    line = plt.vlines(edge, ylim[0], ylim[1], 'k', 'solid', label=f'Edge = {edge} eV')
    line._is_descriptor_line = True


def get_crop(energy: np.ndarray, edge: float, crop_min: int, crop_max: int) -> np.ndarray:
    """
    Get the indices for cropping around the edge.

    Args:
        energy (np.ndarray): Array of energy values.
        edge (float): Energy value of the edge.
        crop_min (int): Minimum crop energy.
        crop_max (int): Maximum crop energy.

    Returns:
        np.ndarray: Indices for cropping the data.
    """
    edge_idx0 = np.argmin(abs(energy - crop_min))
    edge_idx1 = np.argmin(abs(energy - crop_max))
    return np.arange(edge_idx0, edge_idx1)


"""
Single spectrum plotting functions
"""

def plot_spectrum(
    data: Spectrum,
    title: str = '',
    time_averaged: bool = True,
    time_index: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
    remove_labels: bool = False,
    center_on_edge: bool = False,
) -> None:
    """
    Plot a single spectrum.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to ''.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
        remove_labels (bool): Whether to remove labels. Defaults to False.
        center_on_edge (bool): Whether to center the plot on the edge. Defaults to False.
    """
    setup_plot(title)
    
    label = None if remove_labels else getattr(data, 'compound', '__nolegend__')
    crop = get_crop(data.energy, data.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None

    if center_on_edge:
        data = data.copy()
        data.energy -= data.edge

    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, crop, time_index)
    else:
        spectrum_plot(data.energy, data.spectrum, label, crop, time_index)

    finalize_plot()


def plot_spectrum_edge(
    data: Spectrum,
    title: str = 'Zoom to spectrum edge',
    time_averaged: bool = True,
    time_index: Optional[int] = None,
    crop_min: int = 50,
    crop_max: int = 50,
) -> None:
    """
    Plot the spectrum edge for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to 'Zoom to spectrum edge'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (int): Minimum crop around the edge. Defaults to 50.
        crop_max (int): Maximum crop around the edge. Defaults to 50.
    """
    setup_plot(title)
    
    crop = get_crop(data.energy, data.edge, crop_min, crop_max)
    label = getattr(data, 'compound', '__nolegend__')
    
    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, crop, time_index)
    else:
        spectrum_plot(data.energy, data.spectrum, label, crop, time_index)
    
    edge_plot(data.edge)
    finalize_plot()


def plot_spectrum_background(
    data: Spectrum,
    title: str = 'Spectrum background',
    time_averaged: bool = True,
    time_index: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
) -> None:
    """
    Plot the background region for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to 'Spectrum background'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
        time_index (Optional[int]): Which time index to plot. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
    """
    setup_plot(title)
    
    crop = get_crop(data.energy, data.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None
    label = getattr(data, 'compound', '__nolegend__')
    
    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, crop, time_index)
    else:
        spectrum_plot(data.energy, data.spectrum, label, crop, time_index)
        
    label = 'background'
    spectrum_plot(data.energy, data.background, label, crop, time_index)
    finalize_plot()


def plot_spectrum_xanes(
    data: Spectrum,
    title: str = 'XANES region',
    time_averaged: bool = True,
) -> None:
    """
    Plot the XANES region for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to 'XANES region'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
    """
    setup_plot(title)
    
    label = getattr(data, 'compound', '__nolegend__')
    
    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, data.xanes_region_indices)
    else:
        spectrum_plot(data.energy, data.spectrum, label, data.xanes_region_indices)
    
    edge_plot(data.edge)
    finalize_plot()


def plot_spectrum_exafs(
    data: Spectrum,
    title: str = 'EXAFS region',
    time_averaged: bool = True,
) -> None:
    """
    Plot the EXAFS region for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to 'EXAFS region'.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
    """
    setup_plot(title)
    
    label = getattr(data, 'compound', '__nolegend__')
    
    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, data.exafs_region_indices)
    else:
        spectrum_plot(data.energy, data.spectrum, label, data.exafs_region_indices)
    
    edge_plot(data.edge)
    finalize_plot()


def plot_spectrum_descriptor(
    data: Spectrum,
    descriptor: str,
    title: str = 'Spectrum with descriptor',
    crop_min: int = 50,
    crop_max: int = 50,
    time_averaged: bool = True,
) -> None:
    """
    Plot the spectrum with a descriptor for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        descriptor (str): Descriptor to highlight on the plot.
        title (str): Title of the plot. Defaults to 'Spectrum with descriptor'.
        crop_min (int): Minimum crop around the edge. Defaults to 50.
        crop_max (int): Maximum crop around the edge. Defaults to 50.
        time_averaged (bool): Whether to plot the time-averaged spectrum. Defaults to True.
    """
    setup_plot(title)
    
    crop = get_crop(data.energy, data.edge, crop_min, crop_max)
    label = getattr(data, 'compound', '__nolegend__')
    
    if time_averaged:
        spectrum_plot(data.energy, data.time_averaged_spectrum, label, crop)
    else:
        spectrum_plot(data.energy, data.spectrum, label, crop)

    # Get the color of the plotted line for the descriptor
    color = plt.gca().lines[-1].get_color()
    descriptor_plot(getattr(data, descriptor), descriptor, color)

    edge_plot(data.edge)
    finalize_plot()


def plot_spectrum_time_instances(
    data: Spectrum,
    title: str = '',
    time_instance_number: int = 10,
    vertical_displacement_offset: Optional[float] = None,
    time_binning_size: Optional[int] = None,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
    plot_gradient: bool = False,
    cmap: str = 'plasma',
) -> None:
    """
    Plot time instances for a single Spectrum object.

    Args:
        data (Spectrum): The spectrum data.
        title (str): Title of the plot. Defaults to ''.
        time_instance_number (int): Number of time instances to plot. Defaults to 10.
        vertical_displacement_offset (Optional[float]): Offset for vertical displacement. Defaults to None.
        time_binning_size (Optional[int]): Size of the time binning. Defaults to None.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
    """
    setup_plot(title)

    label = getattr(data, 'compound', '__nolegend__')
    crop = get_crop(data.energy, data.edge, crop_min or 0, crop_max or 0) if crop_min or crop_max else None

    if vertical_displacement_offset is None:
        vertical_displacement_offset = data.spectrum[:, 0].mean()

    if time_instance_number is not None:
        if time_instance_number > data.spectrum.shape[1]:
            time_instance_number = data.spectrum.shape[1]
        time_indices = np.linspace(0, data.spectrum.shape[1]-1, time_instance_number).astype(int)
        labels = [f'time {t_idx}' for t_idx in time_indices]
    else: 
        time_indices = None
        labels = None  
        
    if time_binning_size is not None:
        data = data.copy()
        data.bin_time_instances(time_binning_size)
                
    if plot_gradient:
        spectrum_plot_time_gradient(data.energy, data.spectrum, label, crop, time_indices, cmap)
    else:
        spectrum_plot(data.energy, data.spectrum, label, crop, time_indices)
        
        fig = plt.gcf()
        displace_time_instances_vertically(fig, vertical_displacement_offset, labels)


def plot_time_evolution_3d(
    data: Spectrum,
    crop_min: Optional[int] = None,
    crop_max: Optional[int] = None,
    title: str = 'Time Evolution of Spectra',
) -> None:
    """
    Plot the time evolution of spectra in 3D for a single Spectrum object.

    Args:
        data (Spectrum): Data object containing energy, spectra, and time attributes.
        crop_min (Optional[int]): Minimum crop around the edge. Defaults to None.
        crop_max (Optional[int]): Maximum crop around the edge. Defaults to None.
        title (str): Title of the plot. Defaults to 'Time Evolution of Spectra'.
    """
    energy = data.energy
    spectra = np.single(data.spectrum)
    edge = data.edge

    time = np.linspace(0, spectra.shape[1], spectra.shape[1])

    if crop_min is not None and crop_max is not None:
        crop = get_crop(energy, edge, crop_min, crop_max)
        spectra = spectra[crop, :]
        energy = energy[crop]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='w')

    time_grid, energy_grid = np.meshgrid(time, energy)

    ax.plot_surface(
        energy_grid, time_grid, spectra,
        cmap='copper', edgecolor='none',
        linewidth=0.5, alpha=0.8
    )

    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Time')
    ax.set_zlabel('Absorption')
    ax.set_title(title)

    ax.view_init(elev=20, azim=-70)
    plt.show()
