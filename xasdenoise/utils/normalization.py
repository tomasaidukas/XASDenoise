"""
Functions used for XAS spectrum normalization by fitting pre-edge and post-edge regions.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline


class Polynomials:
    """
    Polynomial functions for fitting pre-edge and post-edge regions.
    """

    @staticmethod
    def constant(x, *terms):
        """
        Constant polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Constant values.
        """
        
        return terms[0] * np.ones_like(x)

    @staticmethod
    def linear(x, *terms):
        """
        Linear polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Linear polynomial values.
        """
        
        return terms[0] * x + terms[1]

    @staticmethod
    def quadratic(x, *terms):
        """
        Quadratic polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Quadratic polynomial values.
        """
        
        return terms[0] * x + terms[1] * x**2 + terms[2]

    @staticmethod
    def cubic(x, *terms):
        """
        Cubic polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Cubic polynomial values.
        """
        
        return terms[0] * x + terms[1] * x**2 + terms[2] * x**3 + terms[3]

    @staticmethod
    def quartic(x, *terms):
        """
        Quartic polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Quartic polynomial values.
        """
        
        return terms[0] * x + terms[1] * x**2 + terms[2] * x**3 + terms[3] * x**4 + terms[4]

    @staticmethod
    def victoreen(x, *terms):
        """
        Victoreen polynomial function.

        Args:
            x (np.ndarray): Input values.
            *terms (float): Coefficients.

        Returns:
            np.ndarray: Victoreen polynomial values.
        """
        x = np.maximum(x, 1e-10)  # Set minimum value
        f = 1.23986 * 10**4
        return (terms[0] * f**3 / x**3) - (terms[1] * f**4 / x**4) + terms[2]

class NormFit:
    """
    Class for normalization by fitting pre-edge and post-edge functions.
    """

    def __init__(self):
        self.prelow = 0
        self.prehigh = 0
        self.postlow = 0
        self.posthigh = 0
        self.prefunctional = 0
        self.postfunctional = 0
        self.functional = '0'
        self.e0 = 0
        self.fit_p0_pre_edge = None
        self.fit_p0_post_edge = None
        self.downsample = 0
        self.mask = None

    def norm(self, dataX_fit, dataY_fit, dataY, pre_edge_parameters, post_edge_parameters, e0):
        """
        Normalize data using pre-edge and post-edge fitting.

        Args:
            dataX_fit (np.ndarray): X values for fitting.
            dataY_fit (np.ndarray): Y values for fitting.
            dataY (np.ndarray): Y values for normalization.
            pre_edge_parameters (tuple): Pre-edge fit parameters.
            post_edge_parameters (tuple): Post-edge fit parameters.
            e0 (float): Edge energy.

        Returns:
            array: Normalized Y values.
        """ 
        
        self.e0 = e0
        dataY_pre_fit, self.fit_p0_pre_edge = self._fit_edge(dataX_fit, dataY_fit, pre_edge_parameters, 'pre-edge', self.fit_p0_pre_edge)
        dataY_post_fit, self.fit_p0_post_edge = self._fit_edge(dataX_fit, dataY_fit, post_edge_parameters, 'post-edge', self.fit_p0_post_edge)
        
        index = np.argmin(np.abs(dataX_fit - float(self.e0)))
        edge_jump = (dataY_post_fit - dataY_pre_fit)[index]

        if dataY.ndim == 2:
            NormY = (dataY - dataY_pre_fit[:, None]) / edge_jump
        else:
            NormY = (dataY - dataY_pre_fit) / edge_jump

        dataY_post_fitnorm = (dataY_post_fit - dataY_pre_fit) / edge_jump
        flat_correction = np.zeros_like(dataY_post_fitnorm)
        flat_correction[index:] = 1 - dataY_post_fitnorm[index:]

        if dataY.ndim == 2:
            NormY += flat_correction[:, None]
        else:
            NormY += flat_correction

        return NormY, edge_jump

    def _fit_edge(self, dataX_fit, dataY_fit, edge_parameters, edge_type, p0=None):
        """
        Fit edge region using specified parameters.

        Args:
            dataX_fit (np.ndarray): X values for fitting.
            dataY_fit (np.ndarray): Y values for fitting.
            edge_parameters (tuple): Edge fit parameters.
            edge_type (str): Type of edge ('pre-edge' or 'post-edge').
            p0 (array, optional): Initial guess for fit parameters. Defaults to None.

        Returns:
            tuple: Fitted Y values and optimized parameters.
        """
        
        self._set_edge_parameters(edge_parameters, edge_type)
        dataX_edge, dataY_edge, crop_idx = self.getroi(edge_type, dataX_fit, dataY_fit)
        
        if self.downsample > 0:
            dataX_edge = dataX_edge[::self.downsample]
            dataY_edge = dataY_edge[::self.downsample]
        if self.mask is not None:
            dataY_edge = dataY_edge[self.mask[crop_idx]]
            dataX_edge = dataX_edge[self.mask[crop_idx]]

        dataY_edge_fit, popt = self.regression(dataX_edge, dataY_edge, dataX_fit, p0)
        return dataY_edge_fit, popt

    def regression(self, xregion, yregion, x, p0=None):
        """
        Perform regression to fit data using specified polynomial.

        Args:
            xregion (np.ndarray): X values for fitting.
            yregion (np.ndarray): Y values for fitting.
            x (np.ndarray): Full X range for extrapolation.
            p0 (array, optional): Initial guess for fit parameters. Defaults to None.

        Returns:
            tuple: Fitted Y values and optimized parameters.
        """
        
        if (self.functional == '0') or (self.functional == '0.0'):
            p0 = np.ones(1) if p0 is None or len(p0) != 1 else p0
            popt, _ = curve_fit(Polynomials.constant, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.constant(x, *popt)
        elif (self.functional == '1') or (self.functional == '1.0'):
            p0 = np.ones(2) if p0 is None or len(p0) != 2 else p0
            popt, _ = curve_fit(Polynomials.linear, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.linear(x, *popt)
        elif (self.functional == '2') or (self.functional == '2.0'):
            p0 = np.ones(3) if p0 is None or len(p0) != 3 else p0
            popt, _ = curve_fit(Polynomials.quadratic, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.quadratic(x, *popt)
        elif (self.functional == '3') or (self.functional == '3.0'):
            p0 = np.ones(4) if p0 is None or len(p0) != 4 else p0
            popt, _ = curve_fit(Polynomials.cubic, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.cubic(x, *popt)
        elif (self.functional == '4') or (self.functional == '4.0'):
            p0 = np.ones(5) if p0 is None or len(p0) != 5 else p0
            popt, _ = curve_fit(Polynomials.quartic, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.quartic(x, *popt)
        elif self.functional == 'CS':
            # Fit a smoothing spline
            knots = np.linspace(xregion.min(), xregion.max(), num=self.knots)  # Adjust number of knots
            spline = LSQUnivariateSpline(xregion, yregion, knots[1:-1])  # Adjust scaling of `s`
            yfit = spline(x)
            # from scipy.interpolate import CubicSpline
            # spline_knots = np.linspace(0, len(xregion) - 1, 10).astype(int)
            # cubic_spline = CubicSpline(xregion[spline_knots], yregion[spline_knots])
            # yfit = cubic_spline(x)
            popt = np.zeros(1)

        # elif self.functional == 'V':
        else:
            p0 = np.ones(3) if p0 is None or len(p0) != 3 else p0
            popt, _ = curve_fit(Polynomials.victoreen, xregion, yregion, p0=p0)#, ftol=0.5, xtol=0.5)
            yfit = Polynomials.victoreen(x, *popt)

        return yfit, popt

    def _set_edge_parameters(self, edge_parameters, edge_type):
        """
        Set parameters for edge fitting.

        Args:
            edge_parameters (tuple): Edge fit parameters.
            edge_type (str): Type of edge ('pre-edge' or 'post-edge').
        """
        
        if edge_type == 'pre-edge':
            self.prelow, self.prehigh, self.prefunctional = edge_parameters
            self.functional = str(self.prefunctional)
            self.knots = 8
        elif edge_type == 'post-edge':
            self.postlow, self.posthigh, self.postfunctional = edge_parameters
            self.functional = str(self.postfunctional)
            self.knots = 8

    def getroi(self, region, x, y):
        """
        Get the region of interest for fitting.

        Args:
            region (str): Region type ('pre-edge' or 'post-edge').
            x (np.ndarray): X values.
            y (np.ndarray): Y values.

        Returns:
            tuple: X and Y values in the region of interest, and the indices.
        """
        
        if region == 'pre-edge':
            xllim = float(self.e0) + float(self.prelow)
            xhlim = float(self.e0) + float(self.prehigh)
        elif region == 'post-edge':
            xllim = float(self.e0) + float(self.postlow)
            xhlim = float(self.e0) + float(self.posthigh)

        crop_idx = (x >= xllim) & (x <= xhlim)
        xregion = x[crop_idx]
        yregion = y[crop_idx]
        return xregion, yregion, crop_idx

    def reconstruct_fit(self, x, fit_params, fit_func):
        """
        Reconstruct polynomial fits from stored parameters.
        
        Args:
            x (np.ndarray): Energy values.
            fit_params (np.ndarray): Fitted polynomial parameters.
            fit_func (str): Type of fit function used.
            
        Returns:
            np.ndarray: Reconstructed fit values.
        """
        if fit_func in ['0', '0.0']:
            return Polynomials.constant(x, *fit_params)
        elif fit_func in ['1', '1.0']:
            return Polynomials.linear(x, *fit_params)
        elif fit_func in ['2', '2.0']:
            return Polynomials.quadratic(x, *fit_params)
        elif fit_func in ['3', '3.0']:
            return Polynomials.cubic(x, *fit_params)
        elif fit_func in ['4', '4.0']:
            return Polynomials.quartic(x, *fit_params)
        elif fit_func == 'V':
            return Polynomials.victoreen(x, *fit_params)
        else:
            # Default to linear if function type is unknown
            return Polynomials.linear(x, *fit_params[:2])

    def undo_normalization(self, x, normalized_spectrum, normalization_params):
        """
        Reconstruct unnormalized spectrum from normalized data using stored parameters.
        
        Args:
            x (np.ndarray): Energy values.
            normalized_spectrum (np.ndarray): Normalized spectrum data.
            normalization_params (dict): Dictionary containing normalization parameters.
            
        Returns:
            np.ndarray: Reconstructed unnormalized spectrum.
            
        Raises:
            ValueError: If required normalization parameters are missing.
        """
        if not isinstance(normalization_params, dict):
            raise ValueError("normalization_params must be a dictionary")
            
        required_keys = ['edge', 'pre_edge_region_params', 'post_edge_region_params', 'fit_individual']
        for key in required_keys:
            if key not in normalization_params:
                raise ValueError(f"Missing required parameter: {key}")
        
        edge = normalization_params['edge']
        
        if normalization_params['fit_individual']:
            # Handle individual time instance fits
            if 'individual_params' not in normalization_params:
                raise ValueError("Missing individual_params for individual fitting mode")
                
            reconstructed_spectrum = np.zeros_like(normalized_spectrum)
            
            for time in range(normalized_spectrum.shape[1]):
                time_params = normalization_params['individual_params'][time]
                pre_edge_fit_params = time_params['pre_edge_fit_params']
                post_edge_fit_params = time_params['post_edge_fit_params']
                edge_step = time_params['edge_step']
                
                # Get fit function types
                pre_func = normalization_params['pre_edge_region_params'][2]
                post_func = normalization_params['post_edge_region_params'][2]
                
                # Reconstruct fits
                pre_edge_fit = self.reconstruct_fit(x, pre_edge_fit_params, pre_func)
                post_edge_fit = self.reconstruct_fit(x, post_edge_fit_params, post_func)
                
                # Reconstruct flat correction
                index = np.argmin(np.abs(x - edge))
                post_edge_norm = (post_edge_fit - pre_edge_fit) / edge_step
                flat_correction = np.zeros_like(post_edge_norm)
                flat_correction[index:] = 1 - post_edge_norm[index:]
                
                # Reverse normalization: y_orig = (y_norm - flat_correction) * edge_step + pre_edge_fit
                reconstructed_spectrum[:, time] = (normalized_spectrum[:, time] - flat_correction) * edge_step + pre_edge_fit
                
        else:
            # Handle single normalization for all time instances
            pre_edge_fit_params = normalization_params['pre_edge_fit_params']
            post_edge_fit_params = normalization_params['post_edge_fit_params']
            edge_step = normalization_params['edge_step']
            
            # Get fit function types
            pre_func = normalization_params['pre_edge_region_params'][2]
            post_func = normalization_params['post_edge_region_params'][2]
            
            # Reconstruct fits
            pre_edge_fit = self.reconstruct_fit(x, pre_edge_fit_params, pre_func)
            post_edge_fit = self.reconstruct_fit(x, post_edge_fit_params, post_func)
            
            # Reconstruct flat correction
            index = np.argmin(np.abs(x - edge))
            post_edge_norm = (post_edge_fit - pre_edge_fit) / edge_step
            flat_correction = np.zeros_like(post_edge_norm)
            flat_correction[index:] = 1 - post_edge_norm[index:]
            
            # Reverse normalization
            if normalized_spectrum.ndim == 2:
                reconstructed_spectrum = (normalized_spectrum - flat_correction[:, None]) * edge_step + pre_edge_fit[:, None]
            else:
                reconstructed_spectrum = (normalized_spectrum - flat_correction) * edge_step + pre_edge_fit
        
        return reconstructed_spectrum