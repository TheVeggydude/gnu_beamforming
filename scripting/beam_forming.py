import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c

# Constants
num_samples = 13001
samples_per_second = 100000000
freq_Hz = 1000000.0  # 1 MHz
shift_radians = np.pi/2.0
antenna_distance = 379854120.36127454

# Create the time axis (seconds)
t = np.linspace(0.0, ((num_samples - 1) / samples_per_second), num_samples)

# Create a sine wave, a(t), with a frequency of 1 Hz
a = np.sin((2.0 * np.pi) * freq_Hz * t)

# Create b(t), a (pi / 2.0) phase-shifted replica of a(t)
b = np.sin((2.0 * np.pi) * freq_Hz * t + shift_radians)

# plot the lines
plt.plot(t, a, color="red")
plt.plot(t, b, color="blue")
plt.show()

# Cross-correlate the signals, a(t) & b(t)
ab_corr = np.correlate(a, b, "full")
dt = np.linspace(-t[-1], t[-1], (2 * num_samples) - 1)

# Calculate time & phase shifts
t_shift_alt = (1.0 / samples_per_second) * ab_corr.argmax() - t[-1]
t_shift = dt[ab_corr.argmax()]

# Limit phase_shift to [-pi, pi]
phase_shift = 2.0*np.pi*t_shift/(1/freq_Hz)

manual_t_shift = (shift_radians / (2.0 * np.pi)) / freq_Hz

# Print out applied & calculated shifts
print("Manual time shift: {}".format(manual_t_shift))
print("Alternate calculated time shift: {}".format(t_shift_alt))
print("Calculated time shift: {}".format(t_shift))
print("Manual phase shift: {}".format(shift_radians))
print("Calculated phase shift: {}".format(phase_shift))

# Compute the distance travelled by the longer signal
delta_x = (c / freq_Hz) * (phase_shift / 2 * np.pi)
print(delta_x)

# Compute angle of the source of signal
angle = np.arccos(delta_x/antenna_distance)
print(angle)
