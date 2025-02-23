# -*- coding: utf-8 -*-
"""
Forced_Oscillations

Created on Tue Oct 17 2023

by j01124rd

This code analyzes intensity variation over time, identifying the number 
of oscillations exceeding a specified minimum fractional intensity and 
calculating the time taken to complete these oscillations.

In addition, it provides the time and values of all critical points up
to the breaking point and plots a function with all essential features.
"""
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS

INITIAL_AMPLITUDE = 2.0 # initial amplitude in unit of m^-1

# TIME ARRAY

time = np.arange(0, 2.5, 0.0001)

# INPUT VALUES AND VALIDATIONS

while True:
    frequency = float(input('Enter frequency in the range 1 - 200 Hz = '))
    if 1 < frequency < 200:
        break
    print('Invalid input. Please enter a value between 1 and 200 Hz.')

while True:
    a_1 = float(input('Enter the number in the range 0.1 - 50 m^-1 s^-2 = '))
    if 0.1 < a_1 < 50:
        break
    print('Invalid input. Please enter a value between 0.1 and 50 m^-1 s^-2.')

while True:
    minimum_frac_intensity = float(input(
        'Enter minimum fractional intensity in the range 0 to 1 = ')
        )
    if 0 < minimum_frac_intensity < 1:
        break
    print('Invalid input. Please enter a value between 0 and 1.')

# AMPLITUDE FUNCTION

def amplitude(time, a_1):
    """
    Calculates the amplitude of the function

    Parameters
    ----------
    time : (float)
    a_1 : (float)
    
    Returns
    ----------
    amplitude 
    
    """
    return (1 / (INITIAL_AMPLITUDE + a_1 * time**2)) * np.cos(2 * np.pi * frequency * time)

amplitude_t = amplitude(time, a_1)

# INTENSITY AND MAXIMUM INTENSITY

intensity = np.power(amplitude_t, 2)
max_intensity = np.power(amplitude(0, a_1), 2)

def fractional_intensity(intensity, max_intensity):
    """
    Calculates fractional intensity.

    Parameters
    ----------
    intensity : (float)
    max_intensity : (float)

    Returns
    -------
    fractional_intensity

    """
    return intensity / max_intensity

frac_intensity_values = fractional_intensity(intensity, max_intensity)

# COUNTER

maxima = []
minima = []

MAXIMA_COUNT = -1
MINIMA_COUNT = -1

STOP_COUNT = True
LAST_MINIMA_TIME = 0

# LOOP - analysing peaks; counting until reaches the limit

for i in range(1, len(frac_intensity_values) - 1):
    if STOP_COUNT:
        if (
            frac_intensity_values[i - 1] < frac_intensity_values[i] and
            frac_intensity_values[i] > frac_intensity_values[i + 1]
            ):
            maxima.append((time[i], frac_intensity_values[i]))
            MAXIMA_COUNT += 1
            if frac_intensity_values[i] <= minimum_frac_intensity:
                STOP_COUNT = False
        if LAST_MINIMA_TIME is not None:
            if (
                frac_intensity_values[i - 1] > frac_intensity_values[i] and
                frac_intensity_values[i] < frac_intensity_values[i + 1]
                ):
                minima.append((time[i], frac_intensity_values[i]))
                MINIMA_COUNT += 1
                LAST_MINIMA_TIME = time[i]

# PRINT OUTPUT

# All values for maxima and minima up to the limit

print("All maxima before the limit:")
for point in maxima:
    print(f"Time: {point[0]:.3f} sec, Value: {point[1]:.3f}")

print("All minima before the limit:")
for point in minima:
    print(f"Time: {point[0]:.3f} sec, Value: {point[1]:.3f}")

# Number of oscillations and time at the last minima

print(
      "Number of oscillations before it reaches the limit of minimum fractional intensity = ", 
      MAXIMA_COUNT
      )

if LAST_MINIMA_TIME is not None:
    print(f"Time at the last counted minima: {LAST_MINIMA_TIME:.3f} sec")

# PLOT

plt.figure(figsize=(10, 6))
plt.plot(time, frac_intensity_values, label="Fractional Intensity")
plt.xlabel("Time (sec)")
plt.ylabel("Fractional Intensity")
plt.title("Fractional intensity vs. Time")
plt.grid(True)

# HORIZONTAL LINE - minimum fractional intensity limit with label I_min^f

plt.axhline(
    minimum_frac_intensity, color='green', linestyle='--',
    label=f'Min Intensity ({minimum_frac_intensity})'
    )
plt.text(2.0, minimum_frac_intensity + 0.04, r'$I_{min}^f$', color='green')

# VERTICAL LINE - time at last counted minima with label t_osc

plt.axvline(
    LAST_MINIMA_TIME, color='red', linestyle='--',
    label=f'Time - Last Minima ({LAST_MINIMA_TIME:.3f})'
    )
plt.text(LAST_MINIMA_TIME + 0.04, minimum_frac_intensity + 0.4, r'$t_{osc}$', color='red')

plt.legend()
plt.show()
