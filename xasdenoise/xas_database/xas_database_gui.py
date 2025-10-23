import numpy as np
import plotly.graph_objects as go
from ipywidgets import IntSlider, Button, Dropdown, Checkbox, VBox, HBox, Output, Label, FloatRangeSlider, FloatSlider
from IPython.display import display
from scipy.ndimage import binary_dilation, binary_erosion
from xasdenoise.xas_data import preprocess_spectrum
from xasdenoise.xas_database import processing_utils as preprocess_spectrum_list
import copy
from xasdenoise.utils import normalization
from scipy.ndimage import label

# TODO: upon changing the element, all compounds are still shown, which blocks the plot
# TODO: pre-load only the time-averaged data to avoid having to recompute the time averages every single time
class InteractiveDataProcessing:
    def __init__(self, spectrum_list):
        """
        Initialize the interactive data processing interface with the given spectrum list.

        Args:
            spectrum_list (list): List of spectrum objects to process.
        """
        
        # Initialize data and control variables
        self.input_spectrum_list = spectrum_list
        
        self.selected_elements = np.unique([s.metadata['element'] for s in self.input_spectrum_list])
        self.selected_element = self.selected_elements[0]
        self.selected_monochromator = 'All'
        self.load_element_data()
        
        # Control settings
        self.structuring_element_size = 0
        self.detection_threshold = 95
        self.group_glitches = False
        self.glitch_refinement_fit = False
        self.click_recording_enabled = True
        self.click_count = 0
        self.glitch_fit_models = ['asymmetric_gaussians', 'gaussian', 'skewed_gaussian', 'lorentzian', 'skewed_lorentzian','voigt']
        self.add_to_all_spectra = True  # default mode for adding glitches
        self.glitch_removal_mode = False  # Initialize glitch_removal_mode
        self.add_to_all_spectra = True  # default mode for adding glitches
        self.normalization_ui_on = False
        self.glitch_removal_ui_on = False
        
        # Create Plotly FigureWidget for interactive plotting
        self.fig = go.FigureWidget()
        self.out = Output()

        # Create UI components for both menus
        self.create_main_menu_ui()
        self.create_glitch_removal_ui()
        self.create_normalization_ui()

        # Display the main container and the plot
        self.main_container = self.main_menu_controls
        display(HBox([self.main_container, self.fig]))
        
        self.initialize_plot()

    def create_main_menu_ui(self):
        """
        Create the main menu UI components.
        """
        
        # Create UI layout with ipywidgets
        self.element_dropdown = Dropdown(
            options=self.selected_elements,
            value=self.selected_element,
            description="Element:"
        )
        # self.element_dropdown.observe(self.on_element_change, names='value')
        self.add_observer_once(self.element_dropdown, self.on_element_change)

        # Dropdown for compound selection, with "All" as the first option
        self.monochromator_dropdown = Dropdown(
            options=self.monochromator_options,
            value=self.selected_monochromator,
            description="Monochromator:"
        )
        # self.compound_dropdown.observe(self.on_compound_change, names='value')
        self.add_observer_once(self.monochromator_dropdown, self.on_monochromator_change)
        
        # Dropdown for compound selection, with "All" as the first option
        self.compound_dropdown = Dropdown(
            options=['All'] + self.spectrum_names,
            value=self.spectrum_names[0],
            description="Compound:"
        )
        # self.compound_dropdown.observe(self.on_compound_change, names='value')
        self.add_observer_once(self.compound_dropdown, self.on_compound_change)
        
        # Button to switch to normalization UI
        self.switch_to_normalization_button = Button(
            description="Switch to Normalization", layout={"width": "300px"}
        )
        self.switch_to_normalization_button.on_click(self.show_normalization_ui)

        # Button to switch back to glitch removal UI
        self.switch_to_glitch_removal_button = Button(
            description="Switch to Glitch Removal", layout={"width": "300px"}
        )
        self.switch_to_glitch_removal_button.on_click(self.show_glitch_removal_ui)

        # Main container for switching UIs
        self.main_menu_controls = VBox([Label("Main Menu"),
                                        self.element_dropdown,
                                        self.monochromator_dropdown,
                                        self.compound_dropdown,
                                        self.switch_to_glitch_removal_button,
                                        self.switch_to_normalization_button
                                        ])

    def create_glitch_removal_ui(self):
        """
        Create the UI components for glitch removal functionality.
        """
        
        # Buttons for adding glitch regions
        self.add_region_all_button = Button(description="Add Glitch Mask Regions to All Spectra", layout={'width': '300px'})
        self.add_region_all_button.on_click(self.enable_add_glitch_mode_all)

        self.add_region_single_button = Button(description="Add Glitch Mask Region to Selected Spectrum Only", layout={'width': '300px'})
        self.add_region_single_button.on_click(self.enable_add_glitch_mode_single)

        # Button to remove regions from the glitch mask
        self.remove_region_button = Button(description="Remove Glitch Mask Region from All Spectra", layout={'width': '300px'})
        self.remove_region_button.on_click(self.enable_remove_region_mode_all)
        
        self.select_glitch_region_button = Button(description="Remove Glitch Mask Region from Selected Spectrum Only", layout={'width': '300px'})
        self.select_glitch_region_button.on_click(self.enable_remove_region_mode_single)
        
        # Structuring element size slider
        self.structuring_element_slider = IntSlider(
            value=self.structuring_element_size, min=-500, max=500, description="Structuring Size:", continuous_update=False, layout={'width': '300px'}
        )
        # self.structuring_element_slider.observe(self.update_structuring_element_size, names='value')
        self.add_observer_once(self.structuring_element_slider, self.update_structuring_element_size)

        # Checkboxes for grouping and fitting options in automatic glitch detection
        self.group_glitches_checkbox = Checkbox(value=self.group_glitches, description="Group Glitches", layout={'width': '300px'})
        # self.group_glitches_checkbox.observe(self.on_group_checkbox_change, names='value')
        self.add_observer_once(self.group_glitches_checkbox, self.on_group_checkbox_change)

        self.glitch_fit_checkbox = Checkbox(value=self.glitch_refinement_fit, description="Fit Glitches", layout={'width': '300px'})
        # self.glitch_fit_checkbox.observe(self.on_fit_checkbox_change, names='value')
        self.add_observer_once(self.glitch_fit_checkbox, self.on_fit_checkbox_change)

        # Slider for glitch detection threshold
        self.glitch_detection_threshold_slider = IntSlider(
            value=self.detection_threshold, min=70, max=100, description="Glitch Threshold:", continuous_update=False, layout={'width': '300px'}
        )
        # self.glitch_detection_threshold_slider.observe(self.automatic_glitch_detection, names='value')
        self.add_observer_once(self.glitch_detection_threshold_slider, self.automatic_glitch_detection)

        # Update, Reset, Next Element, and Finish buttons
        self.update_button = Button(description="Update Glitch Mask", layout={'width': '300px'})
        self.update_button.on_click(self.update_glitch_masks)

        self.delete_button = Button(description="Delete Glitch Mask", layout={'width': '300px'})
        self.delete_button.on_click(self.delete_glitch_mask)

        self.reset_button = Button(description="Reset Glitch Mask", layout={'width': '300px'})
        self.reset_button.on_click(self.reset_glitch_mask)

        # Button to apply self.current_glitch_masks to spectrum.glitch_mask
        self.update_spectrum_glitch_mask_button = Button(description="Update Spectrum Glitch Mask", layout={'width': '300px'})
        self.update_spectrum_glitch_mask_button.on_click(self.update_spectrum_glitch_mask)
        
        # Existing glitch removal components
        self.glitch_removal_controls = VBox([
            Label("Main Menu"),
            self.element_dropdown,
            self.monochromator_dropdown,
            self.compound_dropdown,
            self.switch_to_glitch_removal_button,
            self.switch_to_normalization_button,
            
            Label("Glitch Removal Menu"),
            self.add_region_all_button,
            self.add_region_single_button,
            self.remove_region_button,
            self.select_glitch_region_button,
            self.structuring_element_slider,
            self.group_glitches_checkbox,
            self.glitch_fit_checkbox,
            self.glitch_detection_threshold_slider,
            self.update_button,
            self.delete_button,
            self.reset_button,
            self.update_spectrum_glitch_mask_button,
        ])

    def create_normalization_ui(self):
        """
        Create the UI components for spectrum normalization.
        """
        
        # Determine energy range across ALL spectra for this element
        global_min_energy = int(np.min([np.min(e) for e in self.energy]))
        global_max_energy = int(np.max([np.max(e) for e in self.energy]))
        
        # Use representative spectrum for initialization
        idx = self.compound_idx if self.compound_idx is not None else 0
        ref_edge = self.edges[idx]
        ref_energy = self.energy[idx]
        ref_spectrum = self.spectra[idx]
        ref_metadata = self.spectrum_list[idx].metadata
        
        # Check if normalization parameters exist in metadata (from previous normalization)
        has_stored_params = (
            'pre_edge_min_E' in ref_metadata and 
            ref_metadata.get('pre_edge_min_E') is not None
        )
        

        if has_stored_params:
            if abs(ref_metadata['pre_edge_min_E']) == np.inf:
                ref_metadata['pre_edge_min_E'] = np.min(ref_energy)
            if abs(ref_metadata['post_edge_max_E']) == np.inf:
                ref_metadata['post_edge_max_E'] = np.max(ref_energy)
                
                
            # Use stored parameters (convert from edge-relative to absolute)
            pre_edge_min_init = int(ref_edge + ref_metadata['pre_edge_min_E'])
            pre_edge_max_init = int(ref_edge + ref_metadata['pre_edge_max_E'])
            post_edge_min_init = int(ref_edge + ref_metadata['post_edge_min_E'])
            post_edge_max_init = int(ref_edge + ref_metadata['post_edge_max_E'])
            pre_edge_fit_func_init = ref_metadata.get('pre_edge_fit_func', 'V')
            post_edge_fit_func_init = ref_metadata.get('post_edge_fit_func', 'V')
            print(f"Loading stored normalization parameters for {self.spectrum_names[idx]}")
        else:
            # Use intelligent defaults based on minima detection
            pre_edge_min_init = global_min_energy
            pre_edge_max_init = int(self.find_first_minimum_before_edge(ref_energy, ref_spectrum, ref_edge))
            post_edge_min_init = int(self.find_first_minimum_after_edge(ref_energy, ref_spectrum, ref_edge))
            post_edge_max_init = global_max_energy
            pre_edge_fit_func_init = 'V'
            post_edge_fit_func_init = 'V'
            print(f"Using auto-detected normalization parameters for {self.spectrum_names[idx]}")
        
        # Pre-edge minimum slider
        self.normalize_preedge_min = pre_edge_min_init
        self.normalize_preedge_min_slider = IntSlider(
            value=self.normalize_preedge_min, 
            min=global_min_energy, 
            max=ref_edge, 
            description="Pre edge min:", 
            continuous_update=False, 
            layout={'width': '300px'}
        )
        self.add_observer_once(self.normalize_preedge_min_slider, self.update_normalization_params)

        # Pre-edge maximum slider
        self.normalize_preedge_max = pre_edge_max_init
        self.normalize_preedge_max_slider = IntSlider(
            value=self.normalize_preedge_max, 
            min=global_min_energy, 
            max=ref_edge, 
            description="Pre edge max:", 
            continuous_update=False, 
            layout={'width': '300px'}
        )
        self.add_observer_once(self.normalize_preedge_max_slider, self.update_normalization_params)

        # Post-edge minimum slider
        self.normalize_postedge_min = post_edge_min_init
        self.normalize_postedge_min_slider = IntSlider(
            value=self.normalize_postedge_min, 
            min=ref_edge, 
            max=global_max_energy, 
            description="Post edge min:", 
            continuous_update=False, 
            layout={'width': '300px'}
        )
        self.add_observer_once(self.normalize_postedge_min_slider, self.update_normalization_params)

        # Post-edge maximum slider
        self.normalize_postedge_max = post_edge_max_init
        self.normalize_postedge_max_slider = IntSlider(
            value=self.normalize_postedge_max, 
            min=ref_edge, 
            max=global_max_energy, 
            description="Post edge max:", 
            continuous_update=False, 
            layout={'width': '300px'}
        )
        self.add_observer_once(self.normalize_postedge_max_slider, self.update_normalization_params)

        self.pre_edge_fit_checkboxes = []
        self.post_edge_fit_checkboxes = []
        self.pre_edge_fit_func = pre_edge_fit_func_init
        self.post_edge_fit_func = post_edge_fit_func_init
        self.pre_edge_fit_func_selector = Dropdown(
            options=self.pre_edge_fit_funcs, value=self.pre_edge_fit_func, description="Pre-Edge Fit:"
            )
        # self.pre_edge_fit_func_selector.observe(self.update_normalization_params, names="value")
        self.add_observer_once(self.pre_edge_fit_func_selector, self.update_normalization_params)
        
        self.post_edge_fit_func_selector = Dropdown(
            options=self.post_edge_fit_funcs, value=self.post_edge_fit_func, description="Post-Edge Fit:"
            )
        # self.post_edge_fit_func_selector.observe(self.update_normalization_params, names="value")        
        self.add_observer_once(self.post_edge_fit_func_selector, self.update_normalization_params)
        
        self.normalize_button = Button(description="Normalize", layout={'width': '300px'})
        self.normalize_button.on_click(self.normalize)
        
        self.restore_normalization_button = Button(
            description="Restore Original Spectra", 
            layout={'width': '300px'},
            button_style='warning'
        )
        self.restore_normalization_button.on_click(self.restore_normalization)
        
        self.recalculate_mu_button = Button(
            description="Recalculate μ from I0/I1", 
            layout={'width': '300px'},
            button_style='info'
        )
        self.recalculate_mu_button.on_click(self.recalculate_mu)

        # Existing normalization components
        self.normalization_controls = VBox([
            Label("Main Menu"),
            self.element_dropdown,
            self.monochromator_dropdown,
            self.compound_dropdown,
            self.switch_to_glitch_removal_button,
            self.switch_to_normalization_button,
            
            Label("Spectrum Normalization Menu"),
            self.pre_edge_fit_func_selector,
            self.post_edge_fit_func_selector,
            self.normalize_preedge_min_slider,
            self.normalize_preedge_max_slider,
            self.normalize_postedge_min_slider,
            self.normalize_postedge_max_slider,
            self.normalize_button,
            self.restore_normalization_button,
            self.recalculate_mu_button,
        ])

    def add_observer_once(self, widget, callback, event_name="value"):
        """
        Add an observer to a widget only if it hasn't been added already.

        Args:
            widget: The widget to observe.
            callback (function): The callback function to execute when the event is triggered.
            event_name (str): The event to observe. Defaults to "value".
        """
        
        if callback not in widget._trait_notifiers.get(event_name, []):
            widget.observe(callback, names=event_name)

    def find_first_minimum_before_edge(self, energy, spectrum, edge, search_range=50):
        """
        Find the first local minimum in the spectrum before the edge.
        
        Args:
            energy (np.ndarray): Energy array
            spectrum (np.ndarray): Spectrum intensity values
            edge (float): Edge energy
            search_range (float): Energy range to search before edge (default 50 eV)
            
        Returns:
            float: Energy value of the first minimum before edge
        """
        # Find indices in the pre-edge region
        mask = (energy < edge) & (energy > edge - search_range)
        if not np.any(mask):
            # If no points in range, return edge - 10 as default
            return edge - 10
            
        energy_region = energy[mask]
        spectrum_region = spectrum[mask]
        
        # Find local minima using gradient
        gradient = np.gradient(spectrum_region)
        # Find where gradient changes from negative to positive
        minima_indices = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0]
        
        if len(minima_indices) > 0:
            # Return the last (closest to edge) minimum
            return energy_region[minima_indices[-1]]
        else:
            # No clear minimum found, return edge - 10
            return edge - 10

    def find_first_minimum_after_edge(self, energy, spectrum, edge, search_range=100):
        """
        Find the first local minimum in the spectrum after the edge.
        
        Args:
            energy (np.ndarray): Energy array
            spectrum (np.ndarray): Spectrum intensity values
            edge (float): Edge energy
            search_range (float): Energy range to search after edge (default 100 eV)
            
        Returns:
            float: Energy value of the first minimum after edge
        """
        # Find indices in the post-edge region
        mask = (energy > edge) & (energy < edge + search_range)
        if not np.any(mask):
            # If no points in range, return edge + 20 as default
            return edge + 20
            
        energy_region = energy[mask]
        spectrum_region = spectrum[mask]
        
        # Find local minima using gradient
        gradient = np.gradient(spectrum_region)
        # Find where gradient changes from negative to positive
        minima_indices = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0]
        
        if len(minima_indices) > 0:
            # Return the first minimum after edge
            return energy_region[minima_indices[0]]
        else:
            # No clear minimum found, return edge + 20
            return edge + 20
            
    def show_main_menu_ui(self, b=None):
        """
        Display the main menu UI.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.normalization_ui_on = False
        self.glitch_removal_ui_on = False
        self.main_container.children = [self.main_menu_controls]
        self.remove_normalization_lines()

    def show_glitch_removal_ui(self, b=None):
        """
        Display the glitch removal UI.

        Args:
            b: Button triggering the function (optional).
        """
        
        # if not hasattr(self, "glitch_removal_controls"):
        #     self.create_glitch_removal_ui()
        self.normalization_ui_on = False
        self.glitch_removal_ui_on = False
        self.main_container.children = [self.glitch_removal_controls]
        self.remove_normalization_lines()

    def show_normalization_ui(self, b=None):
        """
        Display the normalization UI.

        Args:
            b: Button triggering the function (optional).
        """
        
        # if not hasattr(self, "normalization_controls"):
        #     self.create_normalization_ui()
        self.normalization_ui_on = True
        self.glitch_removal_ui_on = False
        self.main_container.children = [self.normalization_controls]
        # self.create_normalization_ui()
        self.add_normalization_lines()
        
    # --- Data loading and updating functions ---
    def load_element_data(self):
        """
        Load energy, spectra, and glitch masks for the selected element.
        """
        
        # load the data based on the selected element
        self.spectrum_list = preprocess_spectrum_list.get_spectra(self.input_spectrum_list, key='element', value=self.selected_element, copy=False)
            
        # check if monochromator field exists
        monochromators = [spectrum.metadata.get('monochromator', 'Unknown') for spectrum in self.spectrum_list]
        self.monochromator_options = ['All'] + list(np.unique(monochromators))
        
        # also load the data based on the selected monochromator   
        if self.selected_monochromator != 'All':
            self.spectrum_list = preprocess_spectrum_list.get_spectra(self.spectrum_list, key='monochromator', value=self.selected_monochromator, copy=False)
        
        # energy arrays for normalization
        self.energy = [spectrum.energy for spectrum in self.spectrum_list]
        self.min_energy = [np.min(spectrum.energy) for spectrum in self.spectrum_list]
        self.max_energy = [np.max(spectrum.energy) for spectrum in self.spectrum_list]
        self.edges = [spectrum.edge for spectrum in self.spectrum_list]
        
        self.normalize_preedge_range_min = None
        self.normalize_preedge_range_max = None
        self.normalize_postedge_range_min = None
        self.normalize_postedge_range_max = None
        try:
            self.normalize_preedge_range_min = [spectrum.pre_edge_region_indices[0] for spectrum in self.spectrum_list]
            self.normalize_preedge_range_max = [spectrum.pre_edge_region_indices[1] for spectrum in self.spectrum_list]
            self.normalize_postedge_range_min = [spectrum.post_edge_region_indices[0] for spectrum in self.spectrum_list]
            self.normalize_postedge_range_max = [spectrum.post_edge_region_indices[1] for spectrum in self.spectrum_list]
        except:
            pass
        
        self.pre_edge_fit_funcs = ['1', '2', '3', '4', 'V']
        self.post_edge_fit_funcs = ['1', '2', '3', '4', 'V']

        try:
            self.default_pre_edge_fit_func = [self.pre_edge_fit_funcs.index(spectrum.metadata.get('pre_edge_fit_func', self.pre_edge_fit_funcs[0])) for spectrum in self.spectrum_list]
            self.default_post_edge_fit_func = [self.post_edge_fit_funcs.index(spectrum.metadata.get('post_edge_fit_func', self.post_edge_fit_funcs[0])) for spectrum in self.spectrum_list]
        except:
            pass

        
        self.spectra = [spectrum.time_averaged_spectrum for spectrum in self.spectrum_list]
        self.glitch_masks = [np.copy(spectrum.glitch_mask) if spectrum.glitch_mask is not None else np.zeros_like(spectrum.energy, dtype=bool) for spectrum in self.spectrum_list]
        self.spectrum_names = [spectrum.metadata['compound'] for spectrum in self.spectrum_list]
        
        # Store original spectra for restoration
        self.original_spectra = [np.copy(spectrum.spectrum) for spectrum in self.spectrum_list]
        self.original_time_averaged_spectra = [np.copy(spectrum.time_averaged_spectrum) for spectrum in self.spectrum_list]
        
        # Find duplicate names and modify them slightly by appending an index
        name_count = {}
        for i, name in enumerate(self.spectrum_names):
            if name in name_count:
                name_count[name] += 1
                self.spectrum_names[i] = f"{name}_{name_count[name]}"
            else:
               name_count[name] = 0
        
        self.compound_idx = 0
        self.initial_glitch_masks = copy.deepcopy(self.glitch_masks)
        self.current_glitch_masks = copy.deepcopy(self.glitch_masks)
    
    # --- Normalization functions ---
    def update_normalization_params(self, b):
        """
        Update normalization parameters for the selected spectrum or all spectra.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.normalize_preedge_min = self.normalize_preedge_min_slider.value
        self.normalize_preedge_max = self.normalize_preedge_max_slider.value
        self.normalize_postedge_min = self.normalize_postedge_min_slider.value
        self.normalize_postedge_max = self.normalize_postedge_max_slider.value
        self.pre_edge_fit_func = self.pre_edge_fit_func_selector.value
        self.post_edge_fit_func = self.post_edge_fit_func_selector.value
        
        self.add_normalization_lines()

        if self.compound_idx is not None:
            # Single compound selected - preview normalization
            x_fit = self.energy[self.compound_idx].copy()
            y = self.spectra[self.compound_idx].copy()
            edge = self.edges[self.compound_idx]
            data_mask = ~self.current_glitch_masks[self.compound_idx]
            
            self.pre_edge_fit_params = [self.normalize_preedge_min-edge, self.normalize_preedge_max-edge, self.pre_edge_fit_func]
            self.post_edge_fit_params = [self.normalize_postedge_min-edge, self.normalize_postedge_max-edge, self.post_edge_fit_func]
            print("Pre edge fit params: ", self.pre_edge_fit_params)
            print("Post edge fit params: ", self.post_edge_fit_params)
            
            normalise = normalization.NormFit()
            normalise.mask = data_mask
            normalise.downsample = 1
            y, _ = normalise.norm(x_fit, y, y, self.pre_edge_fit_params, self.post_edge_fit_params, edge)
                
            self.update_plot_normalization(y)
        else:
            # "All" selected - preview normalization for all compounds
            print("Previewing normalization for all compounds...")
            # Use the first compound's edge as reference
            ref_edge = self.edges[0]
            self.pre_edge_fit_params = [self.normalize_preedge_min-ref_edge, self.normalize_preedge_max-ref_edge, self.pre_edge_fit_func]
            self.post_edge_fit_params = [self.normalize_postedge_min-ref_edge, self.normalize_postedge_max-ref_edge, self.post_edge_fit_func]
            print(f"Pre edge fit params (relative to edge): {self.pre_edge_fit_params}")
            print(f"Post edge fit params (relative to edge): {self.post_edge_fit_params}")
            
            # Preview normalization for all compounds
            preview_spectra = []
            for idx in range(len(self.spectrum_list)):
                x_fit = self.energy[idx].copy()
                y = self.spectra[idx].copy()
                edge = self.edges[idx]
                data_mask = ~self.current_glitch_masks[idx]
                
                normalise = normalization.NormFit()
                normalise.mask = data_mask
                normalise.downsample = 1
                y_norm, _ = normalise.norm(x_fit, y, y, self.pre_edge_fit_params, self.post_edge_fit_params, edge)
                preview_spectra.append(y_norm)
            
            self.update_plot_normalization_all(preview_spectra)
        
    def normalize(self, b):
        """
        Normalize the selected spectrum or all spectra based on the specified ranges.

        Args:
            b: Button triggering the function (optional).
        """
        
        if self.compound_idx is not None:
            # Single compound selected
            x_fit = self.energy[self.compound_idx]
            y = self.spectrum_list[self.compound_idx].spectrum.copy()
            edge = self.edges[self.compound_idx]
            data_mask = ~self.current_glitch_masks[self.compound_idx]
            
            normalise = normalization.NormFit()
            normalise.mask = data_mask
            normalise.downsample = 1
            for time in range(y.shape[1]):                
                y[:,time], _ = normalise.norm(x_fit, y[:,time], y[:,time], self.pre_edge_fit_params, self.post_edge_fit_params, edge)
            
            self.spectrum_list[self.compound_idx].spectrum = y.copy()
            y = self.spectrum_list[self.compound_idx].time_averaged_spectrum
            self.spectra[self.compound_idx] = y
            
            # Store normalization parameters in existing metadata fields (relative to edge)
            self.spectrum_list[self.compound_idx].metadata['normalized'] = True
            self.spectrum_list[self.compound_idx].metadata['pre_edge_min_E'] = self.pre_edge_fit_params[0]
            self.spectrum_list[self.compound_idx].metadata['pre_edge_max_E'] = self.pre_edge_fit_params[1]
            self.spectrum_list[self.compound_idx].metadata['post_edge_min_E'] = self.post_edge_fit_params[0]
            self.spectrum_list[self.compound_idx].metadata['post_edge_max_E'] = self.post_edge_fit_params[1]
            self.spectrum_list[self.compound_idx].metadata['pre_edge_fit_func'] = self.pre_edge_fit_params[2]
            self.spectrum_list[self.compound_idx].metadata['post_edge_fit_func'] = self.post_edge_fit_params[2]
            
            print("Spectrum normalized successfully.")
            self.add_normalization_lines()
        else:
            # "All" selected - normalize all compounds
            num_compounds = len(self.spectrum_list)
            print(f"Normalizing all {num_compounds} compounds for element {self.selected_element}...")
            
            for idx in range(num_compounds):
                x_fit = self.energy[idx]
                y = self.spectrum_list[idx].spectrum.copy()
                edge = self.edges[idx]
                data_mask = ~self.current_glitch_masks[idx]
                
                normalise = normalization.NormFit()
                normalise.mask = data_mask
                normalise.downsample = 1
                
                for time in range(y.shape[1]):
                    y[:,time], _ = normalise.norm(x_fit, y[:,time], y[:,time], self.pre_edge_fit_params, self.post_edge_fit_params, edge)
                
                # Update the spectrum
                self.spectrum_list[idx].spectrum = y.copy()
                self.spectra[idx] = self.spectrum_list[idx].time_averaged_spectrum
                
                # Store normalization parameters in existing metadata fields (relative to edge)
                self.spectrum_list[idx].metadata['normalized'] = True
                self.spectrum_list[idx].metadata['pre_edge_min_E'] = self.pre_edge_fit_params[0]
                self.spectrum_list[idx].metadata['pre_edge_max_E'] = self.pre_edge_fit_params[1]
                self.spectrum_list[idx].metadata['post_edge_min_E'] = self.post_edge_fit_params[0]
                self.spectrum_list[idx].metadata['post_edge_max_E'] = self.post_edge_fit_params[1]
                self.spectrum_list[idx].metadata['pre_edge_fit_func'] = self.pre_edge_fit_params[2]
                self.spectrum_list[idx].metadata['post_edge_fit_func'] = self.post_edge_fit_params[2]
            
            print(f"Successfully normalized all {num_compounds} compounds.")
            self.initialize_plot()  # Refresh plot to show all normalized spectra

    def restore_normalization(self, b):
        """
        Restore spectra to their original un-normalized state.
        
        Args:
            b: Button triggering the function (optional).
        """
        if self.compound_idx is not None:
            # Restore single compound
            self.spectrum_list[self.compound_idx].spectrum = np.copy(self.original_spectra[self.compound_idx])
            self.spectra[self.compound_idx] = np.copy(self.original_time_averaged_spectra[self.compound_idx])
            
            # Reset normalization metadata to defaults
            self.spectrum_list[self.compound_idx].metadata['normalized'] = False
            # Keep the fields but reset to default values (they're still useful for next normalization)
            # No need to delete them as they're part of the standard metadata structure
            
            print(f"Restored original spectrum for {self.spectrum_names[self.compound_idx]}")
            self.initialize_plot()
        else:
            # Restore all compounds
            for idx in range(len(self.spectrum_list)):
                self.spectrum_list[idx].spectrum = np.copy(self.original_spectra[idx])
                self.spectra[idx] = np.copy(self.original_time_averaged_spectra[idx])
                
                # Reset normalization metadata to defaults
                self.spectrum_list[idx].metadata['normalized'] = False
                # Keep the fields but reset to default values
            
            print(f"Restored original spectra for all {len(self.spectrum_list)} {self.selected_element} compounds")
            self.initialize_plot()

    def recalculate_mu(self, b):
        """
        Recalculate mu (absorption) from I0 and I1 using spectrum.compute_mu().
        This is useful when you want to recompute the absorption after restoring data
        or before re-normalizing.
        
        Args:
            b: Button triggering the function (optional).
        """
        if self.compound_idx is not None:
            # Recalculate mu for single compound
            self.spectrum_list[self.compound_idx].compute_mu()
            # Update the displayed spectrum
            self.spectra[self.compound_idx] = self.spectrum_list[self.compound_idx].time_averaged_spectrum
            
            print(f"Recalculated μ for {self.spectrum_names[self.compound_idx]}")
            self.initialize_plot()
        else:
            # Recalculate mu for all compounds
            for idx in range(len(self.spectrum_list)):
                self.spectrum_list[idx].compute_mu()
                self.spectra[idx] = self.spectrum_list[idx].time_averaged_spectrum
            
            print(f"Recalculated μ for all {len(self.spectrum_list)} {self.selected_element} compounds")
            self.initialize_plot()


    def update_plot_normalization(self, y):
        """
        Update the Plotly plot with the normalized spectrum.

        Args:
            y (array): Normalized spectrum data.
        """
        
        idx = self.compound_idx
        y_glitch = np.where(self.current_glitch_masks[idx], y, np.nan)
        self.fig.data[0].y = y
        self.fig.data[1].y = y_glitch

    def update_plot_normalization_all(self, preview_spectra):
        """
        Update the Plotly plot with all normalized spectra when 'All' is selected.

        Args:
            preview_spectra (list): List of normalized spectrum data arrays.
        """
        # Update all traces in the plot
        trace_idx = 0
        for idx in range(len(self.spectrum_list)):
            y_norm = preview_spectra[idx]
            y_glitch = np.where(self.current_glitch_masks[idx], y_norm, np.nan)
            
            # Update the main spectrum trace
            self.fig.data[trace_idx].y = y_norm
            trace_idx += 1
            # Update the glitch trace
            self.fig.data[trace_idx].y = y_glitch
            trace_idx += 1
            
    def add_normalization_lines(self):
        """
        Add vertical lines for pre-edge and post-edge limits to the plot.
        """
        
        self.remove_normalization_lines()  # Ensure no duplicate lines
        # Add lines for both single compound and "All" selection
        lines = [
            {"name": "pre_min", "x0": self.normalize_preedge_min_slider.value, "x1": self.normalize_preedge_min_slider.value,
            "y0": 0, "y1": 1, "xref": 'x', "yref": 'paper', "line": {"color": "red", "dash": "dot"}},
            {"name": "pre_max", "x0": self.normalize_preedge_max_slider.value, "x1": self.normalize_preedge_max_slider.value,
            "y0": 0, "y1": 1, "xref": 'x', "yref": 'paper', "line": {"color": "red", "dash": "dot"}},
            {"name": "post_min", "x0": self.normalize_postedge_min_slider.value, "x1": self.normalize_postedge_min_slider.value,
            "y0": 0, "y1": 1, "xref": 'x', "yref": 'paper', "line": {"color": "blue", "dash": "dot"}},
            {"name": "post_max", "x0": self.normalize_postedge_max_slider.value, "x1": self.normalize_postedge_max_slider.value,
            "y0": 0, "y1": 1, "xref": 'x', "yref": 'paper', "line": {"color": "blue", "dash": "dot"}},
        ]

        # Update or add lines
        existing_shapes = {shape.name: shape for shape in self.fig.layout.shapes}

        for line in lines:
            if line["name"] in existing_shapes:
                # Update existing line
                shape = existing_shapes[line["name"]]
                shape.x0 = line["x0"]
                shape.x1 = line["x1"]
            else:
                # Add new line
                self.fig.add_shape(line)

    def remove_normalization_lines(self):
        """
        Remove the vertical lines for pre-edge and post-edge limits from the plot.
        """
        
        self.fig.layout.shapes = [shape for shape in self.fig.layout.shapes if shape.name not in {"pre_min", "pre_max", "post_min", "post_max"}]

                
    # --- Glitch removal functions ---            
    def enable_add_glitch_mode_all(self, b=None):
        """
        Enable the mode to add glitch mask regions across all spectra.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.glitch_removal_mode = False
        self.add_to_all_spectra = True
        print("Add glitch region mode for all spectra enabled. Click to set start and end of glitch region.")

    def enable_add_glitch_mode_single(self, b=None):
        """
        Enable the mode to add glitch mask regions for the selected spectrum.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.glitch_removal_mode = False
        self.add_to_all_spectra = False
        print("Add glitch region mode for single spectrum enabled. Click to set start and end of glitch region.")

    def enable_remove_region_mode_all(self, b=None):
        """
        Enable the mode to remove glitch mask regions across all spectra.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.glitch_removal_mode = True
        self.add_to_all_spectra = True
        print("Remove glitch region mode for all spectra enabled. Click to set start and end of glitch region.")

    def enable_remove_region_mode_single(self, b=None):
        """
        Enable the mode to remove glitch mask regions for the selected spectrum.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.glitch_removal_mode = True
        self.add_to_all_spectra = False
        print("Remove glitch region mode for selected spectrum enabled. Click to set start and end of glitch region.")
            
    def update_glitch_masks(self, b):
        """
        Update the glitch masks with the current glitch mask configuration.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.glitch_masks = self.current_glitch_masks.copy()
        print("Glitch mask updated successfully.")

    def delete_glitch_mask(self, b):
        """
        Delete the current glitch masks.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.current_glitch_masks = [np.zeros_like(mask, dtype=bool) for mask in self.current_glitch_masks]
        self.update_plot()
        print("Glitch mask has been deleted.")

    def reset_glitch_mask(self, b):
        """
        Reset the glitch masks to their initial state.

        Args:
            b: Button triggering the function (optional).
        """
        
        self.current_glitch_masks = copy.deepcopy(self.initial_glitch_masks)
        self.update_plot()
        print("Glitch mask has been reset.")
        
    def initialize_plot(self):
        """
        Initialize the interactive plot with the current glitch masks and spectra.
        """
        
        self.fig.data = []  # Clear previous data
        if self.compound_idx is None:  # Plot all spectra if "All" is selected
            for idx, (x, y, mask) in enumerate(zip(self.energy, self.spectra, self.current_glitch_masks)):
                y_glitch = np.where(mask, y, np.nan)
                self.fig.add_trace(go.Line(x=x, y=y, mode='lines', name=self.spectrum_names[idx], opacity=.8))
                # self.fig.add_trace(go.Line(x=x, y=y_glitch, mode='lines', line=dict(color='red', width=4), hoverinfo='skip', opacity=.8))
                self.fig.add_trace(go.Scattergl(x=x, y=y_glitch, mode='markers', marker=dict(color='red', size=4), hoverinfo='skip', opacity=.8))
        else:
            # Plot only the selected spectrum
            idx = self.compound_idx
            x = self.energy[idx]
            y = self.spectra[idx]
            name = self.spectrum_names[idx]
            mask = self.current_glitch_masks[idx]
            y_glitch = np.where(mask, y, np.nan)
            self.fig.add_trace(go.Line(x=x, y=y, mode='lines', name=name, opacity=.8))
            # self.fig.add_trace(go.Line(x=x, y=y_glitch, mode='lines', line=dict(color='red', width=4), hoverinfo='skip', opacity=.8))
            self.fig.add_trace(go.Scattergl(x=x, y=y_glitch, mode='markers', marker=dict(color='red', size=4), hoverinfo='skip', opacity=.8))
            
        # Update layout
        self.fig.update_layout(
            title=f"Glitch Selection for Element {self.selected_element}",
            xaxis_title="Energy (eV)",
            yaxis_title="Absorption",
            showlegend=False,
            width=1200,
            height=1000
        )

        # Set up click events
        for i in range(0, len(self.fig.data), 2):
            self.fig.data[i].on_click(self.on_click)


    def update_plot(self):
        """
        Update the Plotly plot with the current glitch masks and spectra.
        """
        
        if self.compound_idx is None:  # Update all compounds
            for idx, mask in enumerate(self.current_glitch_masks):
                y = self.spectra[idx]
                y_glitch = np.where(mask, y, np.nan)
                self.fig.data[2 * idx].y = y
                self.fig.data[2 * idx + 1].y = y_glitch
        else:  # Update only the selected compound
            idx = self.compound_idx
            y = self.spectra[idx]
            y_glitch = np.where(self.current_glitch_masks[idx], y, np.nan)
            self.fig.data[0].y = y
            self.fig.data[1].y = y_glitch
            
    def on_group_checkbox_change(self, change):
        """
        Handle changes to the "Group Glitches" checkbox.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        self.group_glitches = change['new']
        print(f"Group Glitches: {self.group_glitches}")
        self.automatic_glitch_detection({'new': self.detection_threshold})

    def on_fit_checkbox_change(self, change):
        """
        Handle changes to the "Fit Glitches" checkbox.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        self.glitch_refinement_fit = change['new']
        print(f"Fit Glitches: {self.glitch_refinement_fit}")
        self.automatic_glitch_detection({'new': self.detection_threshold})
    
    def on_normalization_postedge_checkbox_change(self, change):
        """
        Toggle normalization and rerun normalization based on checkbox.
        
        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        self.post_edge_fit_func = change['new']
        print(f"Post edge normalization: {self.post_edge_fit_func}")
        self.update_normalization_params()        
        
    def on_normalization_preedge_checkbox_change(self, change):
        """
        Toggle normalization and rerun normalization based on checkbox.
        
        Args:
            change (dict): Dictionary containing the change event details.
        """
            
        self.pre_edge_fit_func = change['new']
        print(f"Pre edge normalization: {self.pre_edge_fit_func}")
        self.update_normalization_params()        
        
    def on_element_change(self, change):
        """
        Handle changes to the selected element in the dropdown menu.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        print('Element changed')
        self.selected_element = change['new']

        # Suppress compound change during element update
        self.is_updating_compound = True

        # Load element data and update related UI components
        self.selected_monochromator = self.monochromator_options[0]
        self.load_element_data()
        self.compound_dropdown.options = ['All'] + self.spectrum_names  # Update compound options
        self.compound_dropdown.value = self.spectrum_names[0]  # Reset to the first compound
        self.monochromator_dropdown.options = ['All'] + list(np.unique([s.metadata['monochromator'] for s in self.spectrum_list]))
        self.monochromator_dropdown.value = self.monochromator_options[0]
        
        
        # Allow compound changes again
        self.is_updating_compound = False

        # Reinitialize the plot
        self.initialize_plot()
        
        # Reinitialize normalization UI if it is active
        self.create_normalization_ui()
        if self.normalization_ui_on:
            self.show_normalization_ui()
            
    def on_monochromator_change(self, change):
        """
        Handle changes to the selected monochromator in the dropdown menu.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        print('Monochromator changed')
        self.selected_monochromator = change['new']

        # Suppress compound change during monochromator update
        self.is_updating_compound = True

        # Reload element data if monochromator affects it
        self.load_element_data()
        self.compound_dropdown.options = ['All'] + self.spectrum_names  # Update compound options
        self.compound_dropdown.value = self.spectrum_names[0]  # Reset to the first compound

        # Allow compound changes again
        self.is_updating_compound = False

        # Reinitialize the plot
        self.initialize_plot()

        # Reinitialize normalization UI if it is active
        self.create_normalization_ui()
        if self.normalization_ui_on:
            self.show_normalization_ui()
            
    def on_compound_change(self, change):
        """
        Handle changes to the selected compound in the dropdown menu.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        if getattr(self, 'is_updating_compound', False):
            return  # Skip updates if compound change is programmatically triggered

        print('Compound changed')
        selected_value = change['new']

        # Update compound index based on the selected value
        self.compound_idx = None if selected_value == "All" else self.spectrum_names.index(selected_value)

        # Update the plot for the selected compound
        self.initialize_plot()

        # # Reinitialize normalization UI if it is active
        self.create_normalization_ui()
        if self.normalization_ui_on:
            self.show_normalization_ui()
            
        # Only create normalization UI if it doesn't exist yet
        # if not hasattr(self, 'normalization_controls'):
        #     self.create_normalization_ui()
        # elif self.compound_idx is not None:  # If UI exists and a specific compound is selected
        #     # Get the current spectrum and edge values
        #     current_idx = self.compound_idx
        #     current_spectrum = self.spectrum_list[current_idx]
        #     current_edge = self.edges[current_idx] if hasattr(self, 'edges') else None
            
        #     # Update the pre-edge and post-edge function dropdowns to match the current spectrum
        #     if hasattr(self, 'default_pre_edge_fit_func') and hasattr(self, 'pre_edge_fit_func_selector'):
        #         if 0 <= current_idx < len(self.default_pre_edge_fit_func):
        #             default_pre_func = self.pre_edge_fit_funcs[self.default_pre_edge_fit_func[current_idx]]
        #             self.pre_edge_fit_func_selector.value = default_pre_func
        #             self.pre_edge_fit_func = default_pre_func
            
        #     if hasattr(self, 'default_post_edge_fit_func') and hasattr(self, 'post_edge_fit_func_selector'):
        #         if 0 <= current_idx < len(self.default_post_edge_fit_func):
        #             default_post_func = self.post_edge_fit_funcs[self.default_post_edge_fit_func[current_idx]]
        #             self.post_edge_fit_func_selector.value = default_post_func
        #             self.post_edge_fit_func = default_post_func
            
        #     # Temporarily disable observers to prevent triggering update events
        #     is_updating = getattr(self, 'is_updating_normalization', False)
        #     self.is_updating_normalization = True
            
        #     try:
        #         # Update pre-edge region sliders
        #         if hasattr(self, 'normalize_preedge_min_slider') and hasattr(self, 'normalize_preedge_max_slider'):
        #             # Get values from metadata or use defaults
        #             pre_min = current_spectrum.metadata.get('normalize_preedge_min', np.min(self.energy[current_idx]))
        #             pre_max = current_spectrum.metadata.get('normalize_preedge_max', current_edge - 5 if current_edge else np.min(self.energy[current_idx]))
                    
        #             # Update slider ranges first
        #             self.normalize_preedge_min_slider.min = np.min(self.energy[current_idx])
        #             self.normalize_preedge_min_slider.max = current_edge if current_edge else np.max(self.energy[current_idx])
        #             self.normalize_preedge_max_slider.min = np.min(self.energy[current_idx])
        #             self.normalize_preedge_max_slider.max = current_edge if current_edge else np.max(self.energy[current_idx])
                    
        #             # Then update values
        #             self.normalize_preedge_min_slider.value = pre_min
        #             self.normalize_preedge_max_slider.value = pre_max
        #             self.normalize_preedge_min = pre_min
        #             self.normalize_preedge_max = pre_max
                
        #         # Update post-edge region sliders
        #         if hasattr(self, 'normalize_postedge_min_slider') and hasattr(self, 'normalize_postedge_max_slider'):
        #             # Get values from metadata or use defaults
        #             post_min = current_spectrum.metadata.get('normalize_postedge_min', current_edge + 5 if current_edge else np.min(self.energy[current_idx]))
        #             post_max = current_spectrum.metadata.get('normalize_postedge_max', np.max(self.energy[current_idx]))
                    
        #             # Update slider ranges first
        #             self.normalize_postedge_min_slider.min = current_edge if current_edge else np.min(self.energy[current_idx])
        #             self.normalize_postedge_min_slider.max = np.max(self.energy[current_idx])
        #             self.normalize_postedge_max_slider.min = current_edge if current_edge else np.min(self.energy[current_idx])
        #             self.normalize_postedge_max_slider.max = np.max(self.energy[current_idx])
                    
        #             # Then update values
        #             self.normalize_postedge_min_slider.value = post_min
        #             self.normalize_postedge_max_slider.value = post_max
        #             self.normalize_postedge_min = post_min
        #             self.normalize_postedge_max = post_max
        #     finally:
        #         # Restore the original updating state
        #         self.is_updating_normalization = is_updating
            
        #     # Update normalization lines on the plot
        #     self.add_normalization_lines()

        # # Show normalization UI if it's active
        # if self.normalization_ui_on:
        #     self.show_normalization_ui()

    def on_click(self, trace, points, selector):
        """
        Handle user clicks on the plot for glitch region selection.

        Args:
            trace: The trace that was clicked.
            points: Information about the points clicked.
            selector: Selector details.
        """
        
        if not self.click_recording_enabled or len(points.xs) == 0:
            return  # Exit if no points are selected or recording is disabled

        x_click = points.xs[0]

        if self.click_count == 0:
            # First click: record xmin
            self.current_xmin = x_click
            self.click_count = 1
            print(f"First click recorded at x={self.current_xmin}. Waiting for second click to set xmax.")
        else:
            # Second click: record xmax
            self.current_xmax = x_click
            self.click_count = 0  # Reset click count for the next selection
            print(f"Second click recorded at x={self.current_xmax}. X range selected.")

            # Ensure min_x is less than max_x
            self.current_xmin, self.current_xmax = sorted([self.current_xmin, self.current_xmax])

            # Apply the range based on the current button mode
            if self.add_to_all_spectra:
                if self.glitch_removal_mode:
                    # Remove glitch mask from all spectra
                    self.remove_glitch_region_all_spectra(self.current_xmin, self.current_xmax)
                else:
                    # Add glitch mask to all spectra
                    self.add_glitch_region_all_spectra(self.current_xmin, self.current_xmax)
            else:
                # Determine selected_idx based on display mode (single or all compounds)
                if self.compound_idx is None:  # All compounds are displayed
                    selected_idx = points.trace_index // 2  # Each spectrum has two traces: line and scatter
                else:
                    # Single compound is displayed, so use compound index directly
                    selected_idx = self.compound_idx

                if self.glitch_removal_mode:
                    # Remove glitch mask from the selected spectrum
                    self.remove_glitch_region_single_spectrum(selected_idx, self.current_xmin, self.current_xmax)
                else:
                    # Add glitch mask to the selected spectrum
                    self.add_glitch_region_single_spectrum(selected_idx, self.current_xmin, self.current_xmax)

            self.update_plot()


    # Define the actions for adding and removing glitch regions for all/single spectra
    def add_glitch_region_all_spectra(self, xmin, xmax):
        """
        Add glitch mask regions across all spectra in the given x-range.

        Args:
            xmin (float): Start of the x-range.
            xmax (float): End of the x-range.
        """
        
        for idx, energy in enumerate(self.energy):
            mask = self.current_glitch_masks[idx]
            mask |= (energy >= xmin) & (energy <= xmax)
            self.current_glitch_masks[idx] = mask
        print(f"Added glitch region from x=({xmin}, {xmax}) across all spectra.")

    def remove_glitch_region_all_spectra(self, xmin, xmax):
        """
        Remove glitch mask regions across all spectra in the given x-range.

        Args:
            xmin (float): Start of the x-range.
            xmax (float): End of the x-range.
        """
        
        for idx, energy in enumerate(self.energy):
            mask = self.current_glitch_masks[idx]
            mask[(energy >= xmin) & (energy <= xmax)] = False
            self.current_glitch_masks[idx] = mask
        print(f"Removed glitch region from x=({xmin}, {xmax}) across all spectra.")

    def add_glitch_region_single_spectrum(self, spectrum_idx, xmin, xmax):
        """
        Add glitch mask regions for a single spectrum in the given x-range.

        Args:
            spectrum_idx (int): Index of the spectrum.
            xmin (float): Start of the x-range.
            xmax (float): End of the x-range.
        """
        
        mask = self.current_glitch_masks[spectrum_idx]
        mask |= (self.energy[spectrum_idx] >= xmin) & (self.energy[spectrum_idx] <= xmax)
        self.current_glitch_masks[spectrum_idx] = mask
        print(f"Added glitch region from x=({xmin}, {xmax}) in spectrum {spectrum_idx + 1}.")

    def remove_glitch_region_single_spectrum(self, spectrum_idx, xmin, xmax):
        """
        Remove glitch mask regions for a single spectrum in the given x-range.

        Args:
            spectrum_idx (int): Index of the spectrum.
            xmin (float): Start of the x-range.
            xmax (float): End of the x-range.
        """
        
        mask = self.current_glitch_masks[spectrum_idx]
        mask[(self.energy[spectrum_idx] >= xmin) & (self.energy[spectrum_idx] <= xmax)] = False
        self.current_glitch_masks[spectrum_idx] = mask
        print(f"Removed glitch region from x=({xmin}, {xmax}) in spectrum {spectrum_idx + 1}.")

    def update_structuring_element_size(self, change):
        """
        Update the size of the structuring element for morphological operations.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        self.structuring_element_size = change['new']
        modified_masks = [
            self.apply_morphology(mask) for mask in self.glitch_masks
        ]
        self.current_glitch_masks = modified_masks
        self.update_plot()        
        
    def apply_morphology(self, mask):
        """
        Apply morphological operations (dilation/erosion) to a mask.

        Args:
            mask (array): The binary mask to modify.

        Returns:
            array: The modified mask.
        """
        
        if self.structuring_element_size > 0:
            struct_elem = np.ones(self.structuring_element_size, dtype=bool)
            return binary_dilation(mask, structure=struct_elem)
        elif self.structuring_element_size < 0:
            struct_elem = np.ones(abs(self.structuring_element_size), dtype=bool)
            return binary_erosion(mask, structure=struct_elem)
        return mask

    def automatic_glitch_detection(self, change):
        """
        Automatically detect glitches based on the threshold slider value.

        Args:
            change (dict): Dictionary containing the change event details.
        """
        
        self.detection_threshold = change['new']
        modified_masks = []
        if self.compound_idx is None:  # Update all compounds
            for spectrum in self.spectrum_list:
                glitch_mask = preprocess_spectrum.find_glitches(
                    spectrum,
                    threshold=self.detection_threshold,
                    group_glitches=self.group_glitches,
                    glitch_refinement_fit=self.glitch_refinement_fit,
                    glitch_fit_models=self.glitch_fit_models,
                    glitch_fit_max_error=np.inf
                )
                modified_masks.append(glitch_mask)
            
            self.current_glitch_masks = modified_masks
        else:
            # Update only the selected compound
            spectrum = self.spectrum_list[self.compound_idx]
            glitch_mask = preprocess_spectrum.find_glitches(
                spectrum,
                threshold=self.detection_threshold,
                group_glitches=self.group_glitches,
                glitch_refinement_fit=self.glitch_refinement_fit,
                glitch_fit_models=self.glitch_fit_models,
                glitch_fit_max_error=np.inf
            )
            self.current_glitch_masks[self.compound_idx] = glitch_mask
        self.update_plot()
    
    def extract_regions(self, boolean_mask):
        """
        Extract continuous regions from a boolean mask.

        Args:
            boolean_mask (array): 1D array of boolean values.

        Returns:
            list: List of tuples representing start and end indices of regions.
        """
        
        # Label the continuous regions
        
        labeled_array, num_features = label(boolean_mask)

        # Extract start and end indices for each labeled region
        regions = []
        for label_id in range(1, num_features + 1):  # Labels start from 1
            indices = np.where(labeled_array == label_id)[0]
            start, end = indices[0], indices[-1]
            regions.append((start, end))

        return regions
    
    def update_spectrum_glitch_mask(self, b=None):
        """
        Update spectrum glitch masks with the current glitch mask configuration.

        Args:
            b: Button triggering the function (optional).
        """
        
        # Ensure self.current_glitch_masks matches length of self.spectrum_list
        if len(self.current_glitch_masks) != len(self.spectrum_list):
            print("Error: Mismatch between glitch mask count and spectra count.")
            return

        # Update each spectrum's glitch mask
        for spectrum, glitch_mask in zip(self.spectrum_list, self.current_glitch_masks):
            spectrum.glitch_mask = glitch_mask.copy()
            
            # Store glitches as a list of tuples
            spectrum.metadata['glitches'] = []
            if glitch_mask is not None:
                regions = self.extract_regions(glitch_mask.astype(bool))  
                for i, (start, end) in enumerate(regions):
                    spectrum.metadata['glitches'].append((spectrum.energy[start], spectrum.energy[end]))
                
        print("Spectrum glitch masks have been successfully updated.")