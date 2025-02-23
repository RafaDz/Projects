# -*- coding: utf-8 -*-
"""
Assignment: Z Boson

Description:
Data contains information about the 
Z Boson decay into electron and positron pair.

Functions included in this code:
    1. Read combined data files.
    2. Filter unwanted datapoints.
    3. Finds values of two unknown parameters.
    4. Calculates Chi^2 and reduced Chi^2 using minimisation.
    5. Produces fit for the data points with optimized parameters.
    6. Calculates mass and lifetime of Z Boson with uncertainties.
    7. Plots function and data points including important features.
    8. Plots contour diagram representing uncertainties.

Student: j01124rd
"""
# Packages
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.constants import hbar

# Files
#-----------------------------------------------------------------------------

file_names = ['z_boson_data_1.csv', 'z_boson_data_2.csv']

# Input values
#-----------------------------------------------------------------------------
print('---------Input Partial-Width----------')
print('Suggested value (Z_boson decay into e^-e^+ is 83.91 MeV)')
partial_width = float(input('Enter the value of partial width in MeV = ', ))

# Model Function
#-----------------------------------------------------------------------------

def sigma(energy, mass, gamma):
    """
    Model function that represents cross-section in nanobarns.
    It accounts for conversion of partial width into GeV and conversion 
    from natural units into nanobarns

    Parameters
    ----------
    energy (centre of mass) : float
    mass (Z Boson): float
    gamma (lifetime relation): float

    Returns
    -------
    Value of sigma : float

    """
    part_a = (12 * np.pi) / mass**2
    part_b = (energy**2 - mass**2)**2
    part_c = energy**2 / (part_b + mass**2 * gamma**2)
    return part_a * part_c * partial_width**2 * 0.3894

# Read data files and validation
#-----------------------------------------------------------------------------

def read_data(files):
    """
    Reads the data from combined files

    Parameters
    ----------
    files : array

    Returns
    -------
    data_array (string)

    """
    combined_data = []

    for file_name in files:
        try:
            data_array = np.genfromtxt(file_name, delimiter=',',
                                       skip_header=1, dtype='str')
            combined_data.append(data_array)
        except FileNotFoundError:
            print(f"Error: Data file '{file_name}' not found.")

    if combined_data:
        data_array = np.concatenate(combined_data, axis=0)
        return data_array

data_array_combined = read_data(file_names)

# Data filters
#-----------------------------------------------------------------------------

def filter_data(data):
    """
    Filters the data by converting non-numerical 
    strings to NaN and removing outliers.

    Parameters
    ----------
    data : array

    Returns
    -------
    filtered_data : array

    """
    # Replace 'fail' with NaN
    data = np.where(data == 'fail', np.nan, data)
    data = np.where(data == 'error', np.nan, data)

    # Convert the input array to a numeric array
    data = np.array(data, dtype=float)

    # Remove NaN values
    data = data[~np.isnan(data).any(axis=1)]

    # Remove anomalous values in second column
    data = data[data[:, 1] <= 100]
    data = data[data[:, 1] > 0]
    data = data[data[:, 2] != 0]

    return data

filtered_data = filter_data(data_array_combined)

# Initial guesses for parameters
#-----------------------------------------------------------------------------
# Peak of the curve would represent the mass of the particle
# Standard deviation is the width of the Gaussian representing value of gamma

max_index = np.argmax(filtered_data[:, 1])

# First initial guess may not work if anomalous data would not be filtered
# You can then plot it to see what values are around the peak and choose that
# number

initial_guess = [filtered_data[max_index, 0], np.std(filtered_data[:,0])]

# Fit of the function
#-----------------------------------------------------------------------------

def fit_function(centre_of_mass_energy_fit, mass_fit, gamma_fit):
    """
    Model function for fitting.

    Parameters
    ----------
    centre_of_mass_energy_fit : numpy array
        Array of energy values.
    mass_fit : float
        Mass parameter.
    gamma_fit : float
        Gamma parameter.

    Returns
    -------
    numpy array
        Fitted values.
    """
    return sigma(centre_of_mass_energy_fit, mass_fit, gamma_fit)

# Residuals - calculated to remove data greater than 3 standard deviations
#-----------------------------------------------------------------------------

def residuals_function(params, centre_of_mass_energy, data):
    """
    Residuals function for curve fitting.

    Parameters
    ----------
    params : list
        List of parameters [mass, gamma].
    centre_of_mass_energy : numpy array
        Array of energy values.
    data : numpy array
        Array of observed data.

    Returns
    -------
    residuals : numpy array
        Residuals.
    """
    mass, gamma = params
    model_prediction = fit_function(centre_of_mass_energy, mass, gamma)
    residuals = data - model_prediction
    return residuals

# Initial fit function - intermadiate parameters and removal of outliers
#-----------------------------------------------------------------------------

def fit_and_exclude(energy_fit, cross_section_fit, parameters):
    """
    Perform curve fitting and plot the results. 
    Remove the outliers using method of calculating residuals.

    Parameters
    ----------
    energy_fit : numpy array
        Array of energy values.
    cross_section_fit : numpy array
        Array of cross-section values.
    parameters : list
        Initial guess for the fitting parameters.

    Returns
    -------
    intermediate_parameters : numpy array
        Intermediate parameters computed with outliers.
    error_mass_gamma_intermediate : numpy array
        Error on the intermediate parameters.
    filtered_indices : numpy array
        Indices of filtered data.
    """
    intermediate_parameters, intermediate_covariance = \
    curve_fit(sigma, energy_fit, cross_section_fit, p0=parameters)

    error_mass_gamma_intermediate = np.sqrt(np.diag(intermediate_covariance))

    # Filtering outliers three standard devations away from the model function
    residuals = residuals_function(intermediate_parameters,
                                   energy_fit, cross_section_fit)

    filtered_indices = np.abs(residuals) <= 3 * np.std(residuals)

    return intermediate_parameters, error_mass_gamma_intermediate,\
            filtered_indices

intermediate_params, error_on_parameters, filtered_outliers = \
    fit_and_exclude(filtered_data[:, 0], filtered_data[:, 1], initial_guess)

data_without_outliers = filtered_data[filtered_outliers]

energy_no_outliers = data_without_outliers[:, 0]
gamma_no_outliers = data_without_outliers[:, 1]
gamma_uncertainty_no_outliers = data_without_outliers[:, 2]

# Best fit function, minimisation of errors on the parameters
#-----------------------------------------------------------------------------

def best_fit_no_outliers(energy_filtered, gamma_filtered, opt_parameters):
    """
    This function optimizes errors on the initial parameters. 
    It computes values based on filtered data without the outliers.

    Parameters
    ----------
    energy_filtered : float
        Array of filtered energy values.
    gamma_filtered : float
        Array of filtered gamma values.
    opt_parameters: list
        Optimized parameters based on initial fit.

    Returns
    -------
    parameters : numpy array
        Optimized parameters computed without outliers
    error_best : numpy array
        Optimized error on the final parameters
    """
    parameters, covariance = curve_fit(sigma, energy_filtered,
                                 gamma_filtered, p0=opt_parameters)
    error_best = np.sqrt(np.diag(covariance))

    return parameters, error_best

optimal_parameters, minimised_errors = \
    best_fit_no_outliers(energy_no_outliers, gamma_no_outliers,
                         intermediate_params)

optimal_mass = optimal_parameters[0]
optimal_gamma = optimal_parameters[1]

# Chi Squared Optimization
#-----------------------------------------------------------------------------

def chi_squared(params, observation, observation_uncertainty):
    """
    Chi-squared function to be minimized.

    Parameters
    ----------
    params : list
        List of parameters [mass, gamma].
    observation : numpy array of floats
        Observed values.
    observation_uncertainty : numpy array of floats
        Uncertainties on the observed values.

    Returns
    -------
    float
        Chi-squared value.
    """
    mass, gamma = params
    model_prediction = sigma(energy_no_outliers, mass, gamma)
    chi_square = np.sum(((observation - model_prediction) /
                         observation_uncertainty)**2)
    return chi_square

# Perform the minimization
result = minimize(chi_squared, [optimal_mass, optimal_gamma],
                  args=(gamma_no_outliers,
                        gamma_uncertainty_no_outliers))

optimal_mass_fit, optimal_gamma_fit = result.x

min_chi_squared = result.fun

def reduced_chi_squared(minimum_chi):
    """
    Calculates reduced Chi^2 and degrees of freedom.

    Parameters
    ----------
    minimum_chi : float

    Returns
    -------
    reduced_chi_squared : float
    degrees_of_freedom : integer

    """
    deg_of_freedom = len(energy_no_outliers) - len(result.x)
    red_chi_squared = minimum_chi / deg_of_freedom

    return red_chi_squared, deg_of_freedom

reduced_chi_squared, degrees_of_freedom = reduced_chi_squared(min_chi_squared)

# Lifetime and its uncertainty calculations
#-----------------------------------------------------------------------------

def lifetime(gamma):
    """
    Calculate the lifetime of Z Boson.
    Lifetime is calculated from the equation: gamma = hbar/ lifetime.
    It is accounted for the conversion from GeV to SI units.

    Parameters
    ----------
    gamma : float

    Returns
    -------
    lifetime : float
        Lifetime of Z Boson in seconds.
    """
    return hbar / (gamma * 1.602 * 10**(-10))

def lifetime_uncertainty(gamma, gamma_uncertainty):
    """
    Calculate uncertainty of the lifetime of Z Boson.
    It is accounted for the conversion from GeV to SI units.

    Parameters
    ----------
    lifetime : float
    gamma : float
    gamma_uncertainty : float

    Returns
    -------
    lifetime_uncertainty : float

    """
    # Convert gamma_z and gamma_z_uncertainty to SI units
    conversion_gamma = gamma * (1.602 * 10**(-10))
    conversion_gamma_uncertainty = gamma_uncertainty * (1.602 * 10**(-10))

    return (hbar / conversion_gamma**2) * conversion_gamma_uncertainty

lifetime_value = lifetime(optimal_gamma)
uncertainty_value = lifetime_uncertainty(optimal_gamma,
                                         minimised_errors[1])
# Plot
#-----------------------------------------------------------------------------

def plot_filtered_data(centre_of_mass_energy, cross_section, filter_indices,
                       mass_opt, gamma_opt):
    """
    Plot the original data and the fitted curve with filtered points.

    Parameters
    ----------
    centre_of_mass_energy : numpy array
        Array of energy values.
    cross_section : numpy array
        Array of cross-section values.
    filter_indices : numpy array
        Indices of filtered data.
    mass_opt : float
        Optimized mass parameter.
    gamma_opt : float
        Optimized gamma parameter.
    """
    _, axes = plt.subplots(figsize=(8, 5))

    plot_filtered_points(axes, centre_of_mass_energy,
                         cross_section, filter_indices)
    plot_outliers(axes, centre_of_mass_energy, cross_section, filter_indices)
    plot_fitted_curve(axes, centre_of_mass_energy, mass_opt, gamma_opt)
    add_text_box(axes)

    # Add vertical line through the peak of the model function
    axes.axvline(x=mass_opt, color='green', linestyle='--', label='Peak')

    # Add label for the peak value
    axes.text(mass_opt, 0, f'{mass_opt:.2f} 'r'$m_Z$', color='green',
              ha='left', va='bottom', backgroundcolor='white')

    # Add horizontal line representing the width of the function (gamma)
    axes.axhline(y=(max(cross_section) / 2.35),
                 color='purple', linestyle='--',
                 label='Full Width at Half-Maximum')

    # Add label for the full width at half maximum value i.e gamma
    axes.text(90.65, (max(cross_section) / 2.35),
              f'{gamma_opt:.3f} 'r'$\Gamma_Z$', color='purple', ha='left',
              va='bottom', backgroundcolor='white')

    customize_plot(axes)

    plt.savefig('best_fit_z_boson.png', dpi=300)
    plt.show()


def plot_filtered_points(axes, centre_of_mass_energy,
                         cross_section, filter_indices):
    """
    Plot filtered data points.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes on which to plot.
    centre_of_mass_energy : numpy array
        Array of energy values.
    cross_section : numpy array
        Array of cross-section values.
    filter_indices : numpy array
        Indices of filtered data.
    """
    axes.errorbar(centre_of_mass_energy[filter_indices],
                  cross_section[filter_indices],
                  yerr=filtered_data[:,2][filter_indices],
                  fmt='o', label='Filtered Data',
                  ecolor='black',
                  color='#0074D9', markersize=6)


def plot_outliers(axes, centre_of_mass_energy, cross_section, filter_indices):
    """
    Plot outlier data points.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes on which to plot.
    energy_com : numpy array
        Array of energy values.
    cross_section : numpy array
        Array of cross-section values.
    filter_indices : numpy array
        Indices of outlier data.
    """
    axes.scatter(centre_of_mass_energy[~filter_indices],
                 cross_section[~filter_indices], label='Outliers',
                 color='red', marker='x', s=100, linewidths=2)


def plot_fitted_curve(axes, centre_of_mass_energy, mass_opt, gamma_opt):
    """
    Plot the fitted curve.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes on which to plot.
    energy_com : numpy array
        Array of energy values.
    mass_opt : float
        Optimized mass parameter.
    gamma_opt : float
        Optimized gamma parameter.
    """
    energy_range = np.linspace(min(centre_of_mass_energy),
                               max(centre_of_mass_energy), 100)
    fitted_curve = fit_function(energy_range, mass_opt, gamma_opt)
    axes.plot(energy_range, fitted_curve, label='Fitted Curve', color='red')


def add_text_box(axes):
    """
    Add a text box with relevant information.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes on which to add the text box.
    """
    chi_2 = min_chi_squared
    red_chi_2 = reduced_chi_squared
    mass_z = optimal_mass
    gamma_z = optimal_gamma
    mass_z_error = minimised_errors[0]
    gamma_z_error = minimised_errors[1]
    life = lifetime_value
    life_error = uncertainty_value

    text_box_content = (
        f'Min Chi Squared: {chi_2:.3f}\n'
        f'Reduced Chi Squared: {red_chi_2:.3f}\n'
        f'Mass: {mass_z:.2f} ± {mass_z_error:.2g} GeV/c^2\n'
        f'Gamma: {gamma_z:.3f} ± {gamma_z_error:.3f} GeV\n'
        f'Lifetime: {life:.3g} ± {life_error:.3g} s'
    )
    text_box = axes.text(0.75, 0.95, text_box_content,
                         transform=axes.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='white', alpha=1.0))
    _ = text_box


def customize_plot(axes):
    """
    Customize the appearance of the plot.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes to customize.
    """
    axes.set_xlabel('Centre-of-Mass Energy (GeV)')
    axes.set_ylabel('Cross Section (nb)')
    axes.set_title('Observed Data versus Best-fit of the model function')
    axes.grid(True, color='grey', dashes=[2, 2])
    axes.legend(loc='upper left')

plot_filtered_data(filtered_data[:, 0], filtered_data[:, 1], filtered_outliers,
                   optimal_mass, optimal_gamma)

# Contour plot
#-----------------------------------------------------------------------------

def chi_square_contour(mass_parameter, gamma_parameter, data):
    """
    Chi squared loop required to produce contour plot.

    Parameters
    ----------
    mass_parameter : float
    gamma_parameter : float
    data : numpy array

    Returns
    -------
    chi_square : function

    """
    chi_square = 0
    for entry in data:
        chi_square += (((sigma(entry[0], mass_parameter, gamma_parameter)
                        - entry[1]) / entry[2])**2)
    return chi_square

def contour_plot(opt_mass_fit, opt_gamma_fit,
                 min_chi_sq, final_data):
    """
    Producing a contour plot between parameters and obtained 
    minimum chi squared ellipses. Centre is the value of 
    minimum chi squared. 
    Each ellipse represents: 
        Chi^2 + 1.00
        Chi^2 + 2.30
        Chi^2 + 5.99
        Chi^2 + 9.21

    Parameters
    ----------
    opt_mass_fit : float
    opt_gamma_fit : float
    min_chi_squared : float
    filtered_data : numpy array
    """
    mass_values = np.linspace(optimal_mass_fit - 0.045,
                           optimal_mass_fit + 0.045, 500)
    gamma_values = np.linspace(optimal_gamma_fit - 0.045,
                           optimal_gamma_fit + 0.045, 500)

    mass_mesh, gamma_mesh = np.meshgrid(mass_values, gamma_values)
    chi_squared_values = chi_square_contour(mass_mesh, gamma_mesh,
                                            final_data)

    parameters_contour_figure, parameters_contour_plot = plt.subplots()
    _ = parameters_contour_figure

    parameters_contour_plot.set_title(r'$\chi^2_{min}$ contours '
                                      'against parameters.', fontsize=14)
    parameters_contour_plot.set_xlabel(r'$m \quad (GeV/c^2)$', fontsize=14)
    parameters_contour_plot.set_ylabel(r'$\Gamma_Z \quad (GeV)$', fontsize=14)

    parameters_contour_plot.scatter(optimal_mass_fit, optimal_gamma_fit,
                                    label='Minimum')

    chi_squared_levels = [min_chi_sq + 1.00, min_chi_sq + 2.30,
                          min_chi_sq + 5.99, min_chi_sq + 9.21]

    contour_plotting = parameters_contour_plot.contour(mass_mesh, gamma_mesh,
                                                   chi_squared_values,
                                                   levels=chi_squared_levels)

    # Add vertical lines with labels
    delta_mass = minimised_errors[0]
    delta_gamma = minimised_errors[1]

    parameters_contour_plot.axvline(opt_mass_fit + delta_mass,
                                    color='black', linestyle='dashed',
                                    linewidth=1)
    parameters_contour_plot.axvline(opt_mass_fit - delta_mass,
                                    color='black', linestyle='dashed',
                                    linewidth=1)
    parameters_contour_plot.text(opt_mass_fit + delta_mass,
                                 opt_gamma_fit,
                                 r'$m + \Delta m$',
                                 va='bottom', ha='left', color='black',
                                 fontsize=14, weight='bold')
    parameters_contour_plot.text(opt_mass_fit - delta_mass,
                                 opt_gamma_fit,
                                 r'$m - \Delta m$',
                                 va='bottom', ha='right', color='black',
                                 fontsize=14, weight='bold')

    # Add horizontal lines with labels
    parameters_contour_plot.axhline(opt_gamma_fit + delta_gamma,
                                    color='black', linestyle='dashed',
                                    linewidth=1)
    parameters_contour_plot.axhline(opt_gamma_fit - delta_gamma,
                                    color='black', linestyle='dashed',
                                    linewidth=1)
    parameters_contour_plot.text(opt_mass_fit,
                                 opt_gamma_fit + delta_gamma,
                                 r'$\Gamma_Z + '
                                 r'\Delta \Gamma_Z$',
                                 va='bottom', ha='right', color='black',
                                 fontsize=14, weight='bold')
    parameters_contour_plot.text(opt_mass_fit,
                                 opt_gamma_fit - delta_gamma,
                                 r'$\Gamma_Z - '
                                 r'\Delta \Gamma_Z$',
                                 va='top', ha='right', color='black',
                                 fontsize=14, weight='bold')

    box = parameters_contour_plot.get_position()
    parameters_contour_plot.set_position([box.x0, box.y0,
                                          box.width * 1, box.height * 1])

    contour_plotting = parameters_contour_plot.contourf(mass_mesh, gamma_mesh,
                                                    chi_squared_values,
                                                    levels=chi_squared_levels,
                                                    cmap='summer')
    colorbar = plt.colorbar(contour_plotting, ax=parameters_contour_plot,
                            label=r'$\chi^2_{min}$', extend='both')
    _ = colorbar
    contour_plot_lines =\
        parameters_contour_plot.contour(mass_mesh, gamma_mesh,
                                        chi_squared_values,
                                        levels=chi_squared_levels, colors='k',
                                        linewidths=0.2)
    _ = contour_plot_lines

    plt.savefig('contour_plot.png', dpi=300)
    plt.show()

contour_plot(optimal_mass_fit, optimal_gamma_fit, min_chi_squared,
             data_without_outliers)

# Print statements
#-----------------------------------------------------------------------------

print('                                    ')
print('------Chi Squared Optimization------')
print(f'Optimized Parameters: {optimal_mass:.3f}, {optimal_gamma:.3f}')
print(f'Minimum Chi^2 = {min_chi_squared:.3f}')
print(f'Degrees of Freedom = {degrees_of_freedom}')
print(f'Reduced Chi^2 = {reduced_chi_squared:.3f}')
print('                                    ')
print('------Mass and Lifetime of Z Boson------')
print(f'Mass of the Z boson = {optimal_mass:.4g} ± '
      f'{minimised_errors[0]:.2f} GeV/c^2')
print(f'Gamma_Z = {optimal_gamma:.4g} ± '
      f'{minimised_errors[1]:.3f} GeV')
print(f'The lifetime of Z Boson = {lifetime_value:.3g} ± '
      f'{uncertainty_value:.2e} seconds')
