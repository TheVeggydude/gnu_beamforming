import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate

# Constants
shift = - 0.2 * np.pi
noise_amplitude = 0.0
t_max = 20.0
n_samples = 200

# Get time-values
t = np.linspace(0.0, t_max, n_samples)

# Get signal
signal = np.cos(t) + noise_amplitude * np.random.normal(size=(n_samples,))

# Get shifted signal
signal_shifted = np.cos(t+shift) + noise_amplitude * np.random.normal(size=(n_samples,))

# Compute correlation
x_corr = correlate(signal_shifted, signal)

# Recover shift
dt = np.linspace(1-n_samples, n_samples, 2*n_samples-1)
recovered_time_shift = dt[x_corr.argmax()]

recovered_phase_shift = 2 * np.pi * (((0.5 + recovered_time_shift) % 1.0) - 0.5)

# # Spectrum analysis
# spectrum = np.fft.fft(signal_summed_noisy)
# freq = np.fft.fftfreq(len(spectrum))
# threshold = 0.5 * max(abs(spectrum))
# mask = abs(spectrum) > threshold
# peaks = freq[mask]
#
# print(peaks)
#
# plt.plot(freq, abs(spectrum))
# plt.title("Spectrum")
# plt.show()

print(shift, recovered_time_shift, recovered_phase_shift)

# Plot the signals
plt.plot(t, signal, color='blue')
plt.plot(t, signal_shifted, color='red')
plt.title("Signals")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(["Original", "Shifted"])
plt.show()
