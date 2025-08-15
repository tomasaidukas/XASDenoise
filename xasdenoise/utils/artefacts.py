"""
Functions used for automatic glitch detection within XAS data. 

Also contains functions to handle glitch mask creation and loading.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.special import wofz, erf
from scipy.interpolate import UnivariateSpline, interp1d, PchipInterpolator
from scipy.ndimage import label


class GlitchModelFitting:
    """
    Class for fitting and analyzing glitches in data using various models.
    """

    def __init__(self, x, y, models=['gaussian'], glitch_fit_max_width=None, glitch_fit_max_error=None,
                 amplitude=None, center=None, sigma=None, alpha=None, gamma=None,
                 amplitude2=None, center2=None, sigma2=None, alpha2=None, gamma2=None):
        """
        Initialize the glitch fitting class with input data and model parameters.

        Args:
            x (array-like): Energy values of the spectrum.
            y (array-like): Spectrum values to be analyzed for glitches.
            models (list, optional): List of models to fit. Defaults to ['gaussian'].
            glitch_fit_max_width (float, optional): Maximum allowable glitch width. Defaults to None.
            glitch_fit_max_error (float, optional): Maximum allowable fit error. Defaults to None.
            amplitude (float, optional): Amplitude of the primary glitch model. Defaults to None.
            center (float, optional): Center position of the primary glitch model. Defaults to None.
            sigma (float, optional): Standard deviation of the primary glitch model. Defaults to None.
            alpha (float, optional): Skewness parameter for skewed models. Defaults to None.
            gamma (float, optional): Half-width parameter for Lorentzian and Voigt models. Defaults to None.
            amplitude2 (float, optional): Amplitude of the secondary glitch model. Defaults to None.
            center2 (float, optional): Center position of the secondary glitch model. Defaults to None.
            sigma2 (float, optional): Standard deviation of the secondary glitch model. Defaults to None.
            alpha2 (float, optional): Skewness parameter for secondary skewed models. Defaults to None.
            gamma2 (float, optional): Half-width parameter for secondary Lorentzian and Voigt models. Defaults to None.
        """

        # initialize data
        self.x = x
        self.y = y
        self.models = models
        
        # initialize model parameters
        self.amplitude = amplitude
        self.center = center
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        # parameters for the asymmetric models
        self.amplitude2 = amplitude2
        self.center2 = center2
        self.sigma2 = sigma2
        self.alpha2 = alpha2
        self.gamma2 = gamma2
        self.estimate_model_parameters()

        self.available_models = ['gaussian', 'skewed_gaussian', 'asymmetric_skewed_gaussians', 
                                 'lorentzian', 'skewed_lorentzian', 'asymmetric_skewed_lorentzians', 
                                 'voigt', 'skewed_voigt', 'combined_skewed_voigts']
        
        # initialize fitting results
        self.best_fit_model = None
        self.fit_error = None
        self.best_fit_width = None
        self.fit_params = {model: {} for model in models}
        self.glitch_width = 0
        self.glitch_offset = 0
        if glitch_fit_max_width is None:
            self.glitch_fit_max_width = len(self.x)
        else:
            self.glitch_fit_max_width = glitch_fit_max_width
        if glitch_fit_max_error is None:
            self.glitch_fit_max_error = np.inf
        else:
            self.glitch_fit_max_error = glitch_fit_max_error
            
        
    def estimate_model_parameters(self):
        """
        Estimate initial parameters for the glitch models based on the input data.

        Automatically sets attributes for amplitude, center, sigma, alpha, gamma,
        and their secondary counterparts.
        """

        mid = len(self.x) // 2
        
        if self.center is None:    
            self.center = np.argmax((self.y[mid-40:mid+40])) - 40
            # self.center = 0 # the x window array is centered around zero
        if self.amplitude is None:
            # self.amplitude = self.y[self.center + mid] # Detected peak is at the center of the window
            self.amplitude = np.max(self.y[mid-40:mid+40])
        if self.sigma is None:
            self.sigma = 1
        if self.alpha is None:
            self.alpha = 0
        if self.gamma is None:
            self.gamma = 1
        
        # parameters for the asymmetric models
        if self.center2 is None:    
            self.center2 = np.argmin((self.y[mid-40:mid+40])) - 40
            # self.center2 = 0
        if self.amplitude2 is None:
            self.amplitude2 = np.min(self.y[mid-40:mid+40])
            # self.amplitude2 = -self.amplitude
        if self.sigma2 is None:
            self.sigma2 = self.sigma
        if self.alpha2 is None:
            self.alpha2 = 0
        if self.gamma2 is None:
            self.gamma2 = self.gamma
            
    def fit_glitch_models(self):
        """
        Fit all specified models to the spectrum and determine the best-fitting model.

        The method updates:
            - Fit parameters for each model.
            - The best-fitting model based on error.
            - The glitch width and offset of the best-fitting model.
        """
      
        plot = False
        
        if plot:
            plt.figure()
            plt.title("Data being fitted")
            plt.plot(self.x, self.y, label='data')

        for model in self.models:
            self.fit(model)
            self.fit_params[model]['error'] = self.fit_error
            self.fit_params[model]['popt'] = self.popt            
            self.fit_params[model]['glitch_width'] = self.glitch_width
            self.fit_params[model]['glitch_offset'] = self.glitch_offset
            self.fit_params[model]['y_fit'] = self.y_fit
            self.fit_params[model]['fit_failed'] = self.fit_failed
            
            # exclude fits that are too wide to be meaningful
            # if (self.glitch_width is None) or (self.glitch_width > len(self.x)*5):
            if (self.glitch_width is None):
                self.fit_params[model]['error'] = np.inf
                self.fit_params[model]['fit_failed'] = True            
            elif self.fit_error > self.glitch_fit_max_error or (self.glitch_width > self.glitch_fit_max_width):
                self.fit_params[model]['error'] = np.inf
                self.fit_params[model]['fit_failed'] = True
                
            # print(f"Model: {model}, Error: {self.fit_error}, Width: {self.fit_params[model]['glitch_width']}")
            
        # Select the best model based on the fit error
        self.best_fit_model = min(self.fit_params, key=lambda x: self.fit_params[x]['error'])
        if self.fit_params[self.best_fit_model]['fit_failed'] is False:
            self.glitch_width = self.fit_params[self.best_fit_model]['glitch_width']
            self.glitch_offset = self.fit_params[self.best_fit_model]['glitch_offset']
            self.y_fit = self.fit_params[self.best_fit_model]['y_fit']
            
            if plot:
                best_model = self.best_fit_model
                print(f"Best fit model: {best_model}, Error: {self.fit_params[best_model]['error']}, Width: {self.fit_params[best_model]['glitch_width']}")
                # plt.figure()
                plt.title('Best fit model of a glitch: {}'.format(best_model))
                # plt.plot(self.x, self.y, label='data')
                plt.plot(self.x, self.y_fit, '--', label='data fit')
                try:
                    # plt.axvline(x=self.x[len(self.x)//2-int(self.glitch_width)//2], color='red', linestyle='--', linewidth=2)
                    # plt.axvline(x=self.x[len(self.x)//2+int(self.glitch_width)//2], color='red', linestyle='--', linewidth=2)
                    plt.axvline(x=self.x[len(self.x)//2-int(self.glitch_width)//2+int(self.glitch_offset)], color='red', linestyle='--', linewidth=2)
                    plt.axvline(x=self.x[len(self.x)//2+int(self.glitch_width)//2+int(self.glitch_offset)], color='red', linestyle='--', linewidth=2)
                except:
                    pass
        else:
            self.glitch_width = 0
            self.glitch_offset = 0
           
        if plot:
            plt.xlabel('Energy (eV)')
            plt.legend()
            plt.show()
            
    def fit(self, model='gaussian'):
        """
        Fit a specified model to the spectrum data.

        Args:
            model (str): Name of the model to fit. Supported models include
                         'gaussian', 'lorentzian', 'voigt', 'skewed_gaussian', etc.

        Raises:
            ValueError: If the specified model is not supported.
        """

        self.fit_failed = False
        
        try:
            amp_limit = np.inf
            amp2_limit = np.inf
            shift_limit = 400
            shift_limit2 = abs(self.center2) + shift_limit
            
            # fit a model to the data
            if model == 'gaussian':
                p0 = [self.amplitude, self.center, self.sigma]
                bounds_min = [-amp_limit, -shift_limit, 0]
                bounds_max = [amp_limit, shift_limit, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.gaussian
                width_func = GlitchModelFitting.gaussian_width

            elif model == 'shifted_gaussian_tails':
                p0 = [self.amplitude, self.center, self.sigma, self.alpha, 0, 0]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf, -np.inf, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.shifted_gaussian_tails
                width_func = GlitchModelFitting.skewed_gaussian_width

            elif model == 'asymmetric_gaussians':
                p0 = [self.amplitude, self.center, self.sigma, self.amplitude2, self.center2, self.sigma2]
                bounds_min = [-amp_limit, -shift_limit, 0, -amp2_limit, -shift_limit2, 0]
                bounds_max = [amp_limit, shift_limit, np.inf, amp2_limit, shift_limit2, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_gaussians
                width_func = GlitchModelFitting.asymmetric_gaussians_width

            elif model == 'asymmetric_gaussians_flip':
                p0 = [self.amplitude2, self.center2, self.sigma2, self.amplitude, self.center, self.sigma]
                bounds_min = [-amp_limit, -shift_limit, 0, -amp2_limit, -shift_limit2, 0]
                bounds_max = [amp_limit, shift_limit, np.inf, amp2_limit, shift_limit2, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_gaussians
                width_func = GlitchModelFitting.asymmetric_gaussians_width
                
            elif model == 'skewed_gaussian':
                p0 = [self.amplitude, self.center, self.sigma, self.alpha]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.skewed_gaussian
                width_func = GlitchModelFitting.skewed_gaussian_width

            elif model == 'asymmetric_skewed_gaussians':
                p0 = [self.amplitude, self.center, self.sigma, self.alpha, self.amplitude2, self.center2, self.sigma2, self.alpha2]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf, -amp2_limit, -shift_limit2, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, amp2_limit, shift_limit2, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_skewed_gaussians
                width_func = GlitchModelFitting.asymmetric_skewed_gaussians_width

            elif model == 'asymmetric_skewed_gaussians_flip':
                p0 = [self.amplitude2, self.center2, self.sigma2, self.alpha2, self.amplitude, self.center, self.sigma, self.alpha]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf, -amp2_limit, -shift_limit2, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, amp2_limit, shift_limit2, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_skewed_gaussians
                width_func = GlitchModelFitting.asymmetric_skewed_gaussians_width

            elif model == 'lorentzian':
                p0 = [self.amplitude, self.center, self.gamma]
                bounds_min = [-amp_limit, -shift_limit, 0]
                bounds_max = [amp_limit, shift_limit, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.lorentzian
                width_func = GlitchModelFitting.lorentzian_width

            elif model == 'skewed_lorentzian':
                p0 = [self.amplitude, self.center, self.gamma, self.alpha]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.skewed_lorentzian
                width_func = GlitchModelFitting.skewed_lorentzian_width

            elif model == 'asymmetric_skewed_lorentzians':
                p0 = [self.amplitude, self.center, self.gamma, self.alpha, self.amplitude2, self.center2, self.gamma2, self.alpha2]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf, -amp2_limit, -shift_limit2, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, amp2_limit, shift_limit2, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_skewed_lorentzians
                width_func = GlitchModelFitting.asymmetric_skewed_lorentzians_width
                
            elif model == 'asymmetric_lorentzians':
                p0 = [self.amplitude, self.center, self.gamma, self.amplitude2, self.center2, self.gamma2]
                bounds_min = [-amp_limit, -shift_limit, 0, -amp2_limit, -shift_limit2, 0]
                bounds_max = [amp_limit, shift_limit, np.inf, amp2_limit, shift_limit2, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_lorentzians
                width_func = GlitchModelFitting.asymmetric_lorentzians_width

            elif model == 'asymmetric_skewed_lorentzians_flip':
                p0 = [self.amplitude2, self.center2, self.gamma2, self.alpha2, self.amplitude, self.center, self.gamma, self.alpha]
                bounds_min = [-amp_limit, -shift_limit, 0, -np.inf, -amp2_limit, -shift_limit2, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, amp2_limit, shift_limit2, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.asymmetric_skewed_lorentzians
                width_func = GlitchModelFitting.asymmetric_skewed_lorentzians_width

            elif model == 'voigt':
                p0 = [self.amplitude, self.center, self.sigma, self.gamma]
                bounds_min = [-amp_limit, -shift_limit, 0, 0]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.voigt
                width_func = GlitchModelFitting.voigt_width

            elif model == 'skewed_voigt':
                p0 = [self.amplitude, self.center, self.sigma, self.gamma, self.alpha]
                bounds_min = [-amp_limit, -shift_limit, 0, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.skewed_voigt
                width_func = GlitchModelFitting.voigt_width

            elif model == 'combined_skewed_voigts':
                p0 = [self.amplitude, self.center, self.sigma, self.gamma, self.alpha,
                    self.amplitude2, self.center2, self.sigma2, self.gamma2, self.alpha2]
                bounds_min = [-amp_limit, -shift_limit, 0, 0, -np.inf, -amp2_limit, -shift_limit2, 0, 0, -np.inf]
                bounds_max = [amp_limit, shift_limit, np.inf, np.inf, np.inf, amp2_limit, shift_limit2, np.inf, np.inf, np.inf]
                bounds = (bounds_min, bounds_max)
                fit_func = self.combined_skewed_voigts
                width_func = GlitchModelFitting.voigt_width

            else:
                raise ValueError(f"Model '{model}' is not supported.")

            # do the fitting
            self.popt, _ = optimize.curve_fit(fit_func, self.x, self.y, p0=p0, bounds=bounds)
            # self.popt, _ = optimize.curve_fit(fit_func, self.x, self.y, p0=p0)
            self.y_fit = fit_func(self.x, *self.popt)
            self.fit_error = np.sqrt(np.mean((self.y - self.y_fit)**2))            
            self.glitch_width = abs(width_func(self.popt))
            self.glitch_offset = self.popt[1]
            
        except RuntimeError as e:
            # print(f"RuntimeError during fitting with model '{model}': {e}")
            self.fit_error = np.inf  # Assign a high error value to indicate a poor fit
            self.popt = None
            self.y_fit = None
            self.fit_failed = True
            
            self.glitch_width = None
            self.glitch_offset = None
    
    def plot_all_models(self):
        """
        Plot all available glitch models for visualization and comparison.
        """
      
        for model in self.available_models:
            if model == 'gaussian':
                y = self.gaussian(self.x, self.amplitude, self.center, self.sigma)
            elif model == 'skewed_gaussian':
                y = self.skewed_gaussian(self.x, self.amplitude, self.center, self.sigma, self.alpha)
            elif model == 'asymmetric_skewed_gaussians':
                y = self.asymmetric_skewed_gaussians(self.x, self.amplitude, self.center, self.sigma, self.alpha,
                                                             self.amplitude2, self.center2, self.sigma2, self.alpha2)
            elif model == 'lorentzian':
                y = self.lorentzian(self.x, self.amplitude, self.center, self.gamma)
            elif model == 'skewed_lorentzian':
                y = self.skewed_lorentzian(self.x, self.amplitude, self.center, self.gamma, self.alpha)
            elif model == 'asymmetric_skewed_lorentzians':
                y = self.asymmetric_skewed_lorentzians(self.x, self.amplitude, self.center, self.gamma, self.alpha,
                                                               self.amplitude2, self.center2, self.gamma2, self.alpha2)
            elif model == 'asymmetric_lorentzians':
                y = self.asymmetric_lorentzians(self.x, self.amplitude, self.center, self.gamma,
                                                               self.amplitude2, self.center2, self.gamma2)
            elif model == 'voigt':
                y = self.voigt(self.x, self.amplitude, self.center, self.sigma, self.gamma)
            elif model == 'skewed_voigt':
                y = self.skewed_voigt(self.x, self.amplitude, self.center, self.sigma, self.gamma, self.alpha)
            elif model == 'combined_skewed_voigts':
                y = self.combined_skewed_voigts(self.x, self.amplitude, self.center, self.sigma, self.gamma, self.alpha,
                                                        self.amplitude2, self.center2, self.sigma2, self.gamma2, self.alpha2)
            plt.figure()
            plt.title(f'{model} glitch model')
            plt.plot(self.x, y, label=model)
            plt.xlabel('Energy (eV)')
            plt.legend()
            plt.show()
       
    @staticmethod
    def gaussian_width(fit_params):
        """
        Compute the width of a Gaussian model.

        Args:
            fit_params (array-like): Parameters of the Gaussian model. Expected format:
                [amplitude, center, sigma].

        Returns:
            float: The width of the Gaussian (6 * sigma, covering ±3 standard deviations).
        """
        sigma = abs(fit_params[2])
        return 6 * sigma


    @staticmethod
    def skewed_gaussian_width(fit_params):
        """
        Compute the width of a skewed Gaussian model.

        Args:
            fit_params (array-like): Parameters of the skewed Gaussian model. Expected format:
                [amplitude, center, sigma, alpha].

        Returns:
            float: Approximate width of the skewed Gaussian, adjusted for skewness.
        """
        sigma = abs(fit_params[2])
        alpha = abs(fit_params[3])
        return (6 + abs(alpha)) * sigma


    @staticmethod
    def asymmetric_skewed_gaussians_width(fit_params):
        """
        Compute the width of an asymmetric skewed Gaussian model.

        Args:
            fit_params (array-like): Parameters of the asymmetric skewed Gaussian model. Expected format:
                [amplitude1, center1, sigma1, alpha1, amplitude2, center2, sigma2, alpha2].

        Returns:
            float: Combined width of the asymmetric skewed Gaussian, including separation between peaks.
        """
        sigma1 = abs(fit_params[2])
        sigma2 = abs(fit_params[6])
        alpha1 = abs(fit_params[3])
        alpha2 = abs(fit_params[7])
        separation = abs(fit_params[1]) + abs(fit_params[5])
        width1 = (6 + abs(alpha1)) * sigma1 / 2
        width2 = (6 + abs(alpha2)) * sigma2 / 2
        return width1 + width2 + separation


    @staticmethod
    def asymmetric_gaussians_width(fit_params):
        """
        Compute the width of an asymmetric Gaussian model.

        Args:
            fit_params (array-like): Parameters of the asymmetric Gaussian model. Expected format:
                [amplitude1, center1, sigma1, amplitude2, center2, sigma2].

        Returns:
            float: Combined width of the two Gaussian components, including separation between peaks.
        """
        sigma1 = abs(fit_params[2])
        sigma2 = abs(fit_params[4])
        separation = abs(fit_params[1]) + abs(fit_params[3])
        width1 = 6 * sigma1 / 2
        width2 = 6 * sigma2 / 2
        return width1 + width2 + separation


    @staticmethod
    def lorentzian_width(fit_params):
        """
        Compute the width of a Lorentzian model.

        Args:
            fit_params (array-like): Parameters of the Lorentzian model. Expected format:
                [amplitude, center, gamma].

        Returns:
            float: The width of the Lorentzian (8 * gamma, covering the full width at 1/10 maximum).
        """
        gamma = abs(fit_params[2])
        return 8 * gamma


    @staticmethod
    def skewed_lorentzian_width(fit_params):
        """
        Compute the width of a skewed Lorentzian model.

        Args:
            fit_params (array-like): Parameters of the skewed Lorentzian model. Expected format:
                [amplitude, center, gamma, alpha].

        Returns:
            float: Approximate width of the skewed Lorentzian, adjusted for skewness.
        """
        gamma = abs(fit_params[2])
        alpha = abs(fit_params[3])
        return (8 + abs(alpha)) * gamma


    @staticmethod
    def asymmetric_skewed_lorentzians_width(fit_params):
        """
        Compute the width of an asymmetric skewed Lorentzian model.

        Args:
            fit_params (array-like): Parameters of the asymmetric skewed Lorentzian model. Expected format:
                [amplitude1, center1, gamma1, alpha1, amplitude2, center2, gamma2, alpha2].

        Returns:
            float: Combined width of the asymmetric skewed Lorentzian, including separation between peaks.
        """
        gamma1 = abs(fit_params[2])
        gamma2 = abs(fit_params[6])
        alpha1 = abs(fit_params[3])
        alpha2 = abs(fit_params[7])
        separation = abs(fit_params[1]) + abs(fit_params[5])
        width1 = (8 + abs(alpha1)) * gamma1 / 2
        width2 = (8 + abs(alpha2)) * gamma2 / 2
        return width1 + width2 + separation


    @staticmethod
    def asymmetric_lorentzians_width(fit_params):
        """
        Compute the width of an asymmetric Lorentzian model.

        Args:
            fit_params (array-like): Parameters of the asymmetric Lorentzian model. Expected format:
                [amplitude1, center1, gamma1, amplitude2, center2, gamma2].

        Returns:
            float: Combined width of the two Lorentzian components, including separation between peaks.
        """
        gamma1 = abs(fit_params[2])
        gamma2 = abs(fit_params[4])
        separation = abs(fit_params[1]) + abs(fit_params[3])
        width1 = 8 * gamma1 / 2
        width2 = 8 * gamma2 / 2
        return width1 + width2 + separation


    @staticmethod
    def voigt_width(fit_params):
        """
        Compute the width of a Voigt model.

        Args:
            fit_params (array-like): Parameters of the Voigt model. Expected format:
                [amplitude, center, sigma, gamma].

        Returns:
            float: Approximate width of the Voigt model, combining Gaussian and Lorentzian components.
        """
        sigma = abs(fit_params[2])
        gamma = abs(fit_params[3])
        return 6 * sigma / 2 + 8 * gamma / 2


    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        """
        Gaussian-shaped glitch model.

        Args:
            x (array-like): Input values (e.g., energy values).
            amplitude (float): Amplitude of the Gaussian component.
            center (float): Center of the Gaussian component.
            sigma (float): Standard deviation of the Gaussian component.

        Returns:
            array-like: The values of the Gaussian glitch for each input value in x.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


    @staticmethod
    def skewed_gaussian(x, amplitude, center, sigma, alpha):
        """
        Skewed Gaussian model for asymmetric peak.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude (float): Amplitude of the Gaussian peak.
            center (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
            alpha (float): Skewness parameter.

        Returns:
            array-like: Values of the skewed Gaussian at each point in x.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) * (1 + erf(alpha * (x - center) / (np.sqrt(2) * sigma)))


    @staticmethod
    def shifted_gaussian_tails(x, amplitude, center, sigma, alpha, B_L, B_R):
        """
        Skewed Gaussian with additional shifted tails.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude (float): Amplitude of the Gaussian peak.
            center (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
            alpha (float): Skewness parameter.
            B_L (float): Left tail shift magnitude.
            B_R (float): Right tail shift magnitude.

        Returns:
            array-like: Values of the skewed Gaussian with tails at each point in x.
        """
        def sigmoid(x, slope=0.01):
            return 1 / (1 + np.exp(-slope * x))

        gaussian = GlitchModelFitting.skewed_gaussian(x, amplitude, center, sigma, alpha)
        left_tail_shift = B_L * (1 - sigmoid(x - center))
        right_tail_shift = B_R * sigmoid(x - center)
        return gaussian + left_tail_shift + right_tail_shift


    @staticmethod
    def asymmetric_gaussians(x, amplitude1, mu1, sigma1, amplitude2, mu2, sigma2):
        """
        Combined model using two Gaussian functions to represent an asymmetric dip and rise.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude1 (float): Amplitude of the first Gaussian peak.
            mu1 (float): Mean of the first Gaussian.
            sigma1 (float): Standard deviation of the first Gaussian.
            amplitude2 (float): Amplitude of the second Gaussian peak.
            mu2 (float): Mean of the second Gaussian.
            sigma2 (float): Standard deviation of the second Gaussian.

        Returns:
            array-like: Combined values of the two Gaussians at each point in x.
        """
        return GlitchModelFitting.gaussian(x, amplitude1, mu1, sigma1) + GlitchModelFitting.gaussian(x, amplitude2, mu2, sigma2)


    @staticmethod
    def asymmetric_skewed_gaussians(x, amplitude1, mu1, sigma1, alpha1, amplitude2, mu2, sigma2, alpha2):
        """
        Combined model using two skewed Gaussian functions to represent an asymmetric dip and rise.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude1 (float): Amplitude of the first skewed Gaussian peak.
            mu1 (float): Mean of the first Gaussian.
            sigma1 (float): Standard deviation of the first Gaussian.
            alpha1 (float): Skewness parameter for the first Gaussian.
            amplitude2 (float): Amplitude of the second skewed Gaussian peak.
            mu2 (float): Mean of the second Gaussian.
            sigma2 (float): Standard deviation of the second Gaussian.
            alpha2 (float): Skewness parameter for the second Gaussian.

        Returns:
            array-like: Combined values of the two skewed Gaussians at each point in x.
        """
        return GlitchModelFitting.skewed_gaussian(x, amplitude1, mu1, sigma1, alpha1) + GlitchModelFitting.skewed_gaussian(x, amplitude2, mu2, sigma2, alpha2)


    @staticmethod
    def lorentzian(x, amplitude, center, gamma):
        """
        Lorentzian-shaped glitch model.

        Args:
            x (array-like): Input values (e.g., energy values).
            amplitude (float): Amplitude of the Lorentzian component.
            center (float): Center of the Lorentzian component.
            gamma (float): Half-width at half-maximum (HWHM) of the Lorentzian component.

        Returns:
            array-like: The values of the Lorentzian glitch for each input value in x.
        """
        return (amplitude * gamma**2) / ((x - center)**2 + gamma**2)


    @staticmethod
    def skewed_lorentzian(x, amplitude, center, gamma, alpha):
        """
        Skewed Lorentzian function to model an asymmetric peak with broad tails.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude (float): Amplitude of the peak.
            center (float): Center of the Lorentzian peak.
            gamma (float): Half-width at half-maximum (HWHM) for the Lorentzian.
            alpha (float): Skewness parameter; positive values skew to the right, negative values to the left.

        Returns:
            array-like: Values of the skewed Lorentzian at each point in x.
        """
        lorentzian = GlitchModelFitting.lorentzian(x, amplitude, center, gamma)
        skew_factor = 1 + erf(alpha * (x - center))
        return lorentzian * skew_factor


    @staticmethod
    def asymmetric_skewed_lorentzians(x, amplitude1, mu1, gamma1, alpha1, amplitude2, mu2, gamma2, alpha2):
        """
        Combined model using two skewed Lorentzians functions to represent an asymmetric dip and rise.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude1 (float): Amplitude of the first skewed Lorentzian peak.
            mu1 (float): Center of the first Lorentzian peak.
            gamma1 (float): HWHM for the first Lorentzian.
            alpha1 (float): Skewness parameter for the first Lorentzian.
            amplitude2 (float): Amplitude of the second skewed Lorentzian peak.
            mu2 (float): Center of the second Lorentzian peak.
            gamma2 (float): HWHM for the second Lorentzian.
            alpha2 (float): Skewness parameter for the second Lorentzian.

        Returns:
            array-like: Combined values of the two skewed Lorentzians at each point in x.
        """
        return GlitchModelFitting.skewed_lorentzian(x, amplitude1, mu1, gamma1, alpha1) + GlitchModelFitting.skewed_lorentzian(x, amplitude2, mu2, gamma2, alpha2)


    @staticmethod
    def asymmetric_lorentzians(x, amplitude1, mu1, gamma1, amplitude2, mu2, gamma2):
        """
        Combined model using two Lorentzian functions to represent an asymmetric dip and rise.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude1 (float): Amplitude of the first Lorentzian peak.
            mu1 (float): Center of the first Lorentzian peak.
            gamma1 (float): HWHM for the first Lorentzian.
            amplitude2 (float): Amplitude of the second Lorentzian peak.
            mu2 (float): Center of the second Lorentzian peak.
            gamma2 (float): HWHM for the second Lorentzian.

        Returns:
            array-like: Combined values of the two Lorentzians at each point in x.
        """
        return GlitchModelFitting.lorentzian(x, amplitude1, mu1, gamma1) + GlitchModelFitting.lorentzian(x, amplitude2, mu2, gamma2)


    @staticmethod
    def voigt(x, amplitude, center, sigma, gamma):
        """
        Voigt-shaped glitch model (convolution of Gaussian and Lorentzian).

        Args:
            x (array-like): Input values (e.g., energy values).
            amplitude (float): Amplitude of the Voigt component.
            center (float): Center of the Voigt component.
            sigma (float): Standard deviation of the Gaussian component.
            gamma (float): HWHM of the Lorentzian component.

        Returns:
            array-like: The values of the Voigt glitch for each input value in x.
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


    @staticmethod
    def skewed_voigt(x, amplitude, center, sigma, gamma, alpha):
        """
        Skewed Voigt profile function to model an asymmetric peak with long tails.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude (float): Amplitude of the peak.
            center (float): Center of the peak.
            sigma (float): Gaussian standard deviation (controls peak sharpness).
            gamma (float): Lorentzian HWHM (controls tails).
            alpha (float): Skewness parameter.

        Returns:
            array-like: Values of the skewed Voigt profile at each point in x.
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        voigt = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        skew_factor = 1 + erf(alpha * (x - center))
        return voigt * skew_factor


    @staticmethod
    def combined_skewed_voigts(x, amplitude1, center1, sigma1, gamma1, alpha1, amplitude2, center2, sigma2, gamma2, alpha2):
        """
        Combined model using two skewed Voigt profiles to represent an asymmetric peak and dip.

        Args:
            x (array-like): Independent variable (e.g., energy values).
            amplitude1, amplitude2 (float): Amplitudes of the skewed Voigt profiles.
            center1, center2 (float): Centers of the skewed Voigt profiles.
            sigma1, sigma2 (float): Gaussian standard deviations for each Voigt profile.
            gamma1, gamma2 (float): Lorentzian HWHMs for each Voigt profile.
            alpha1, alpha2 (float): Skewness parameters for each Voigt profile.

        Returns:
            array-like: Combined values of the two skewed Voigt profiles at each point in x.
        """
        voigt1 = GlitchModelFitting.skewed_voigt(x, amplitude1, center1, sigma1, gamma1, alpha1)
        voigt2 = GlitchModelFitting.skewed_voigt(x, amplitude2, center2, sigma2, gamma2, alpha2)
        return voigt1 + voigt2

    
def does_spectrum_have_glitches(data, tol=50):
    """
    Determine if the absorption spectrum contains monochromator glitches.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        tol (int, optional): Tolerance for matching glitch positions. Defaults to 50.

    Returns:
        bool: True if glitches are detected, otherwise False.
    """

    I0 = data.time_averaged_I0
    mu = data.time_averaged_spectrum
    x = data.energy
    
    # ignore data around the edge
    exclude = (x > (data.first_minima_before_edge-20)) * (x < (data.first_maxima_after_edge+20))

    # extract glitch positions
    glitch_positions_I0, _ = estimate_glitch_positions(x, I0, exclude)
    glitch_positions_mu, _ = estimate_glitch_positions(x, mu, exclude)
    
    # glitch positions will be +- some indices shifted with respect to each other
    # we can find the difference between the two sets of glitch positions
    # and find the matching glitches within some difference tolerance such as 5 pixel shifts
    glitch_positions, glitch_indices = correlate_glitch_positions(glitch_positions_I0, glitch_positions_mu, tol)
    matching_glitches = len(glitch_positions)
    
    # print("Number of matching glitches {} out of {}".format(matching_glitches, max([len(glitch_positions_I0), len(glitch_positions_mu)])))    
    if matching_glitches > 0:
        print("Spectrum {} potentially has glitches".format(data.metadata['compound']))
    #     plt.figure()
    #     plt.title("Potential glitches in {}".format(data.metadata['compound']))
    #     plt.plot(x[glitch_positions],mu[glitch_positions],'ro')     
    #     plt.plot(x,mu)
    #     plt.xlabel('Energy (eV)')
    #     plt.show()
    return matching_glitches > 0
    
    # NOTE: this is also not bad but it is not very reliable
    # correlation of I0 and mu gradients should be high if there are glitches
    # corr_mu = abs(pearsonr(mu_grad, I0_grad)[0])
    # return corr_mu > threshold
    
def correlate_glitch_positions(glitch_positions1, glitch_positions2, tol=10):
    """
    Correlate glitch positions from two signals to find matching glitches within a tolerance.

    Args:
        glitch_positions1 (array-like): Glitch positions from the first signal.
        glitch_positions2 (array-like): Glitch positions from the second signal (reference).
        tol (int, optional): Tolerance for matching glitch positions. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - glitch_positions (array-like): Matched glitch positions.
            - glitch_indices (array-like): Indices of the matched glitches.
    """

    glitch_positions = []
    glitch_indices = []
    for glitch_ref in glitch_positions2:
        match = sum((abs(glitch_positions1 - glitch_ref) < tol))
        if match > 0:
            glitch_idx = np.round(np.mean(np.where(abs(glitch_positions1 - glitch_ref) < tol)[0])).astype(int)
            glitch_indices.append(glitch_idx)
            
            glitch_pos = glitch_positions1[glitch_idx]
            glitch_positions.append(glitch_pos)
            # glitch_positions.append(np.arange(-tol//2,tol//2) + glitch_pos)
    return glitch_positions, glitch_indices
            
            
def estimate_glitch_positions(x, y, mask=None, percentile=95, group_glitches=False, process=True):
    """
    Estimate the positions of glitches in a signal.

    Args:
        x (array-like): Energy values of the spectrum.
        y (array-like): Spectrum containing glitches.
        mask (array-like, optional): Mask for excluding regions from glitch detection. Defaults to None.
        percentile (int, optional): Percentile value for thresholding the signal. Defaults to 95.
        group_glitches (bool, optional): Whether to group glitches close to each other. Defaults to False.
        process (bool, optional): Whether to process the data to find glitches. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - glitch_positions (array-like): Indices of the glitch positions.
            - glitch_widths (array-like): Estimated widths of the glitches.
    """

    # process the data to find the glitches    
    if process:
        y_proc = gaussian_filter1d(y, sigma=5)    
        # gradient
        y_proc = abs(np.roll(y_proc, 1) - y_proc)
        y_proc[0] = y_proc[1]
        # y_proc = abs(np.roll(y_proc, 1) - y_proc)
        # y_proc[0] = y_proc[1]
        
    # mask regions from glitch detection
    if mask is not None:
        x_notmasked = x
        x = x[~mask]
        y = y[~mask]
        y_proc = y_proc[~mask]
        
    # Estimate threshold which separates glitches from the signal
    threshold0 = np.percentile(y_proc, percentile)
    threshold1 = np.percentile(y_proc, 100)
    threshold = np.array([threshold0, threshold1])
    glitch_positions = signal.find_peaks(y_proc, height=threshold)[0]
    glitch_widths = signal.peak_widths(y_proc, glitch_positions, rel_height=.8)[0]
    

    # Avoid cutting out too much data by limiting the number of peaks
    # note: one peak does not equal one glitch
    # max_peak_num = 100
    # if len(glitch_positions) > max_peak_num:
    #     # Find peaks in the smoothness array above the threshold
    #     # try to increase the threshold if too many peaks are found
    #     for tries in range(10):
    #         if len(glitch_positions) > max_peak_num:
    #             threshold = threshold * 1.5
    #             glitch_positions = signal.find_peaks(y_proc, height=threshold)[0]
    #             glitch_widths = signal.peak_widths(y_proc, glitch_positions, rel_height=.8)[0]
    #         else:
    #             break
    
    # find glitches close to each other and assume they are the same glitch    
    if group_glitches:
        position_tol = np.min(np.abs(np.diff(np.sort(glitch_positions))))*2    
        glitch_positions_new = []
        glitch_widths_new = []
        
        while len(glitch_positions) > 0:
            glitch = glitch_positions[0]
            idx = np.where(abs(glitch_positions - glitch) < position_tol)[0]
            glitch_positions_new.append(np.round(np.mean(glitch_positions[idx])))
            
            # width is computed as half_glitch_width + peak_to_peak_distance + half_glitch_width
            width = glitch_widths[idx[0]]//2 + (glitch_positions[idx[-1]] - glitch_positions[idx[0]]) + glitch_widths[idx[-1]]//2
            glitch_widths_new.append(width)
            # glitch_widths_new.append(np.mean(glitch_widths[idx]))
            
            # remove indices that were matches from the search
            glitch_positions = np.delete(glitch_positions, idx)
            glitch_widths = np.delete(glitch_widths, idx)
        
        glitch_widths = np.array(glitch_widths_new).astype(int)
        glitch_positions = np.array(glitch_positions_new).astype(int)
    
    # extrapolate the masked indices to the non-masked index range
    if mask is not None:
        tmp = np.zeros_like(glitch_positions)
        for idx, glitch in enumerate(glitch_positions):
            tmp[idx] = np.where(x_notmasked == x[glitch])[0]
        glitch_positions = tmp
     
    return glitch_positions, glitch_widths


def estimate_glitch_mask(x, glitch_positions, glitch_widths):
    """
    Create a mask for regions containing glitches.

    Args:
        x (array-like): Energy values of the spectrum.
        glitch_positions (array-like): Glitch positions.
        glitch_widths (array-like): Glitch widths.

    Returns:
        array-like: Boolean mask for glitch regions.
    """

    # for the glitch indices by fitting a gaussian and using the stdev as the width of the glitch
    glitch_mask = np.zeros(len(x), dtype=bool)
    
    # loop over each glitch peak and fit a gaussian to it
    for glitch_pos, glitch_width in zip(glitch_positions, glitch_widths):
        
        # extract the coordinates of the glitch based on fit parameters
        x0 = round(glitch_pos - glitch_width)
        if x0 < 0: x0 = 0
        x1 = round(glitch_pos + glitch_width)
        if x1 >= len(x): x1 = len(x)-1
        
        glitch_mask[x0:x1] = True

    return glitch_mask

def estimate_glitch_mask_fit(x, y, glitch_positions, glitch_widths, glitch_width_scaling=1,
                             glitch_fit_window=1000, glitch_fit_max_width=None, glitch_fit_max_error=np.inf,
                             fit_models=['gaussian', 'skewed_gaussian', 'asymmetric_skewed_gaussians',
                                     'lorentzian', 'skewed_lorentzian', 'asymmetric_skewed_lorentzians',
                                     'shifted_gaussian_tails', 'voigt']):
    """
    Refine the glitch mask by fitting models to each glitch.

    Args:
        x (array-like): Energy values of the spectrum.
        y (array-like): Spectrum containing glitches.
        glitch_positions (array-like): Glitch positions.
        glitch_widths (array-like): Glitch widths.
        glitch_width_scaling (float, optional): Scaling factor for glitch width. Defaults to 1.
        glitch_fit_window (int, optional): Window size for fitting glitches. Defaults to 1000.
        glitch_fit_max_width (float, optional): Maximum allowable glitch width. Defaults to None.
        glitch_fit_max_error (float, optional): Maximum allowable fit error. Defaults to infinity.
        fit_models (list, optional): List of models to fit. Defaults to common glitch models.

    Returns:
        array-like: Boolean mask for refined glitch regions.
    """
    
    # for the glitch indices by fitting a gaussian and using the stdev as the width of the glitch
    glitch_mask = np.zeros(len(y), dtype=bool)
    
    # loop over each glitch peak and fit a gaussian to it
    for glitch_pos, glitch_width in zip(glitch_positions, glitch_widths):
        # TODO: better window width estimation is possible
        # define the cropping window
        # glitch_fit_window = 1000
        # or use an adaptive window size based on the estimated glitch width
        # this is typically too narrow
        if glitch_fit_window is None:
            glitch_fit_window = max([100,int(glitch_width)*2])

        # take a window around every peak    
        pts = glitch_fit_window
        xmin = glitch_pos - pts//2
        xmax = glitch_pos + pts//2
        if xmax >= len(y):
            pts = len(y) - glitch_pos
        elif xmin < 0:
            pts = glitch_pos
            
        # crop a window around the peak for fitting        
        x_fit = np.arange(-pts//2,pts//2) 
        y_fit = y[x_fit + glitch_pos]      
        
        # if the region is already masked, skip the fitting
        # also skip fitting if more than half of the region is masked
        # mask_fit = glitch_mask[x_fit + glitch_pos]
        # if sum(mask_fit) > len(mask_fit)//2:
        #     x_fit = x_fit[~mask_fit]
        #     y_fit = y_fit[~mask_fit]
            
        # remove a linear trend from the data
        # coefs = np.polyfit(x_fit, y_fit, 1)
        # y_fit = y_fit - np.polyval(coefs, x_fit)
        # remove a polynomial trend from the data (use edges of the window to avoid including the glitch)
        crop = np.concatenate((np.arange(0,pts//4), np.arange(3*pts//4,pts)))
        # crop = np.concatenate((np.arange(0,pts//5), np.arange(4*pts//5,pts)))
        coefs = np.polyfit(x_fit[crop], y_fit[crop], 2)
        y_fit = y_fit - np.polyval(coefs, x_fit)
               
        # normalize
        y_fit = (y_fit - np.min(y_fit)) / (np.max(y_fit) - np.min(y_fit))
        y_fit = (y_fit - np.mean(y_fit)) / np.std(y_fit)
        y_fit = y_fit / np.max(y_fit)
        
        # Fit models to the data
        glitch_fitting = GlitchModelFitting(x_fit, y_fit, models=fit_models, glitch_fit_max_width=glitch_fit_max_width, glitch_fit_max_error=glitch_fit_max_error)
        glitch_fitting.fit_glitch_models()
        glitch_width = glitch_fitting.glitch_width//2 * glitch_width_scaling
        glitch_offset = glitch_fitting.glitch_offset
        
        if glitch_width > 0:
            # extract the coordinates of the glitch based on fit parameters
            x0 = round(glitch_pos + glitch_offset - glitch_width)
            if x0 < 0: x0 = 0
            x1 = round(glitch_pos + glitch_offset + glitch_width)
            if x1 >= len(y): x1 = len(y)-1
            
            glitch_mask[x0:x1] = True
    return glitch_mask

def estimate_baseline(x, y, data_mask=None, plot=False):
    """
    Estimate the baseline of the spectrum, excluding glitch regions.

    Args:
        x (array-like): Energy values of the spectrum.
        y (array-like): Spectrum values.
        data_mask (array-like, optional): Mask to exclude glitch regions. Defaults to None.
        plot (bool, optional): Whether to plot the baseline fitting results. Defaults to False.

    Returns:
        array-like: Estimated baseline of the spectrum.
    """

    # Fit a baseline to the data excluding the glitch regions
    if data_mask is None:
        data_mask = np.zeros(len(y), dtype=bool)
    baseline_mask = ~data_mask
    
    from pybaselines import Baseline, utils
    baseline_fitter = Baseline(x_data=x)
    
    baseline, params = baseline_fitter.poly(y, weights=baseline_mask, poly_order=3)
    # baseline, params = baseline_fitter.penalized_poly(y, weights=baseline_mask, poly_order=3)
    # baseline, params = baseline_fitter.iasls(y, weights=baseline_mask)
    # baseline = baseline_fitter.pspline_asls(y, weights=baseline_mask)[0]

    if plot:
        plt.figure()
        plt.suptitle('Baseline fitting and glitch removal')
        plt.subplot(1,2,1)
        plt.plot(x, y, label='data')
        plt.plot(x[data_mask], y[data_mask], 'go', alpha=0.1, label='exclude from fitting')  
        plt.xlabel('Energy (eV)')  
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(x, y, label='data')
        plt.plot(x, baseline, label='baseline')  
        plt.xlabel('Energy (eV)')  
        plt.legend()
    return baseline

def detect_glitches(x, y, mask=None, threshold=95, glitch_width_scaling=1, group_glitches=False, plot=False):
    """
    Detect glitches in the spectrum using peak detection and optional grouping.

    Args:
        x (array-like): Energy values of the spectrum.
        y (array-like): Spectrum containing glitches.
        mask (array-like, optional): Mask to exclude certain regions. Defaults to None.
        threshold (int, optional): Percentile threshold for detecting glitches. Defaults to 95.
        glitch_width_scaling (float, optional): Scaling factor for glitch widths. Defaults to 1.
        group_glitches (bool, optional): Whether to group glitches close to each other. Defaults to False.
        plot (bool, optional): Whether to plot the glitch detection results. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - glitch_positions (array-like): Indices of detected glitches.
            - glitch_widths (array-like): Estimated widths of the glitches.
            - glitch_mask (array-like): Boolean mask for glitch regions.
    """

    # locate the glitches
    glitch_positions, glitch_widths = estimate_glitch_positions(x, y, mask=mask, percentile=threshold, group_glitches=group_glitches)
    if plot:            
        plt.figure()
        plt.suptitle('Glitch position detection')
        plt.scatter(x[glitch_positions], y[glitch_positions], color='blue', label='glitch locations')    
        plt.plot(x, y, label='spectrum')
        plt.xlabel('Energy (eV)')        
        plt.legend()
        plt.show()
    
    # estimate the glitch mask based on the glitch positions and their widths from the scipy peak finder
    glitch_mask = estimate_glitch_mask(x, glitch_positions, glitch_widths*glitch_width_scaling)
    if plot:
        plt.figure()
        plt.suptitle('Glitch detection')
        plt.plot(x[glitch_mask], y[glitch_mask], 'r.', label='glitches')  
        plt.plot(x, y, label='data')
        plt.xlabel('Energy (eV)')  
        plt.legend()
        plt.show()    
    
    return glitch_positions, glitch_widths, glitch_mask

def detect_matching_glitches(x, y1, y2, mask=None, threshold=95, glitch_width_scaling=1, group_glitches=False, tol=10, plot=False):
    """
    Detect matching glitches in two spectra by correlating glitch positions.

    The function identifies glitches in two spectra (e.g., I0 and mu) and correlates their positions
    to determine matching glitches. It uses peak detection and optional grouping to estimate glitch
    positions and widths.

    Args:
        x (array-like): The energy array.
        y1 (array-like): The first spectrum (e.g., I0).
        y2 (array-like): The second spectrum (e.g., mu).
        mask (array-like, optional): Boolean mask to exclude regions from glitch detection. Defaults to None.
        threshold (int, optional): Percentile threshold for glitch detection. Defaults to 95.
        glitch_width_scaling (float, optional): Scaling factor for glitch widths. Defaults to 1.
        group_glitches (bool, optional): Whether to group glitches that are close to each other. Defaults to False.
        tol (int, optional): Tolerance for matching glitch positions between the two spectra. Defaults to 10.
        plot (bool, optional): Whether to plot the glitch detection results. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - glitch_positions (array-like): Indices of matching glitches.
            - glitch_widths (array-like): Estimated widths of matching glitches.
            - glitch_mask (array-like): Boolean mask indicating the positions of matching glitches.
    """

    # locate the glitches
    glitch_positions1, glitch_widths1 = estimate_glitch_positions(x, y1, mask=mask, percentile=threshold, group_glitches=group_glitches)
    glitch_positions2, glitch_widths2 = estimate_glitch_positions(x, y2, mask=mask, percentile=threshold, group_glitches=group_glitches)
    glitch_positions, glitch_indices = correlate_glitch_positions(glitch_positions1, glitch_positions2, tol=tol)
    glitch_widths = glitch_widths1[glitch_indices]
    
    if plot:            
        plt.figure()
        plt.suptitle('Glitch position detection')
        plt.subplot(1,2,1)
        plt.scatter(x[glitch_positions], y1[glitch_positions], color='blue', label='glitch locations')    
        plt.plot(x, y1, label='spectrum')
        plt.xlabel('Energy (eV)')      
        plt.legend()  
        plt.subplot(1,2,2)
        plt.scatter(x[glitch_positions], y2[glitch_positions], color='blue', label='glitch locations')    
        plt.plot(x, y2, label='spectrum')
        plt.xlabel('Energy (eV)')        
        plt.legend()
        plt.show()
    
    # estimate the glitch mask based on the glitch positions and their widths from the scipy peak finder
    glitch_mask = estimate_glitch_mask(x, glitch_positions, glitch_widths*glitch_width_scaling)
    if plot:
        plt.figure()
        plt.suptitle('Glitch detection')
        plt.subplot(1,2,1)
        plt.plot(x[glitch_mask], y1[glitch_mask], 'r.', label='glitches')  
        plt.plot(x, y1, label='data')
        plt.xlabel('Energy (eV)')  
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(x[glitch_mask], y2[glitch_mask], 'r.', label='glitches')  
        plt.plot(x, y2, label='data')
        plt.xlabel('Energy (eV)')  
        plt.legend()
        plt.show()    
    
    return glitch_positions, glitch_widths, glitch_mask

def remove_glitches(x, y, glitch_mask, plot=False, mode='interp'):
    """
    Remove or correct regions containing glitches in the spectrum.

    Args:
        x (array-like): Energy values of the spectrum.
        y (array-like): Spectrum containing glitches.
        glitch_mask (array-like): Boolean mask for glitch regions.
        plot (bool, optional): Whether to plot the glitch removal results. Defaults to False.
        mode (str, optional): Mode of glitch removal. Options are:
                              'interp', 'interp_avg', 'pchip', 'spline', 'delete'.
                              Defaults to 'interp'.

    Returns:
        tuple: A tuple containing:
            - x (array-like): Energy values after glitch removal.
            - y (array-like): Corrected spectrum values.
    """

    data_mask = ~glitch_mask
    
    if y.ndim == 1:
        y = y[:,np.newaxis]
        
    y0 = np.mean(y,axis=1)
    
    if mode == 'delete':
        y = y[data_mask, :]
        x = x[data_mask]
    else:
        if mode == 'interp_avg':
            # use smoothed and averaged data for interpolation to avoid noise effects
            y_avg = np.mean(y, axis=1)
            y_avg = gaussian_filter1d(y_avg, sigma=5)
            
            # f = interp1d(x[data_mask], y_avg[data_mask], kind='slinear',
            #             bounds_error=False, axis=0, fill_value=(y_avg[0],y_avg[-1]))
            f = PchipInterpolator(x[data_mask], y_avg[data_mask], axis=0)
            y_avg = f(x)
            
            # replace glitch regions from the averaged and interpolated array
            y[glitch_mask,:] = y_avg[glitch_mask, np.newaxis]
            
        elif mode == 'interp':
            # scipy.interp1d - interp an N-dimensional array along an axis, good for qexafs data:
            f = interp1d(x[data_mask], y[data_mask,:], kind='slinear',
                        bounds_error=False, axis=0, fill_value=(y[0,:],y[-1,:]))
            y = f(x)
        
        elif mode == 'pchip':
            f = PchipInterpolator(x[data_mask], y[data_mask, :], axis=0)
            y = f(x)
        
        elif mode == 'spline':
            for t in range(y.shape[1]):
                # UnivariateSpline - smooth spline interpolation
                spline = UnivariateSpline(x[data_mask], y[data_mask,t], s=0, k=3)
                y[:,t] = spline(x)
            
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Glitch removal')
        plt.plot(x[glitch_mask], y0[glitch_mask], 'r.', alpha=0.3, label='glitches')    
        plt.plot(x, y0, label='spectrum')
        plt.xlabel('Energy (eV)')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(x, np.mean(y,axis=1), label='interpolated spectrum')
        plt.xlabel('Energy (eV)')
        plt.legend()
        plt.show()
    
    if y.shape[1] == 1:
        y = y[:,0]
        
    return x, y

def extract_glitch_region_indices(boolean_mask):
    """
    Extract start and end indices for continuous glitch regions.

    Args:
        boolean_mask (array-like): Boolean mask indicating glitch regions.

    Returns:
        list: List of tuples containing start and end indices for each glitch region.
    """

    # Label the continuous regions
    if np.sum(boolean_mask) == 0:
        return []
    else:
        labeled_array, num_features = label(boolean_mask)

        # Extract start and end indices for each labeled region
        regions = []
        for label_id in range(1, num_features + 1):  # Labels start from 1
            indices = np.where(labeled_array == label_id)[0]
            start, end = indices[0], indices[-1]
            regions.append((start, end))

        return regions

def extract_glitch_region_energies(energy, boolean_mask):
    """
    Extract start and end energy values for continuous glitch regions.

    Args:
        energy (array-like): Energy values of the spectrum.
        boolean_mask (array-like): Boolean mask indicating glitch regions.

    Returns:
        list: List of tuples containing start and end energy values for each glitch region.
    """

    # Label the continuous regions
    if np.sum(boolean_mask) == 0:
        return []
    else:
        labeled_array, num_features = label(boolean_mask)
        # Extract start and end indices for each labeled region
        regions = []
        for label_id in range(1, num_features + 1):  # Labels start from 1
            indices = np.where(labeled_array == label_id)[0]
            start, end = indices[0], indices[-1]
            
            regions.append((energy[start], energy[end]))

        return regions

def construct_glitch_mask_from_regions(energy, glitch_regions):
    """
    Construct a glitch mask from a list of glitch regions.

    Args:
        energy (array-like): Energy values of the spectrum.
        glitch_regions (list): List of tuples containing start and end energies for glitch regions.

    Returns:
        array-like: Boolean mask for glitch regions.
    """

    glitch_mask = np.zeros(len(energy), dtype=bool)
    for start, end in glitch_regions:
        glitch_idx = (energy >= start) & (energy <= end)
        glitch_mask[glitch_idx] = True
        
    return glitch_mask