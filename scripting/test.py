import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, random
from scipy.signal import correlate

period = 1.0  # period of oscillations (seconds)
tmax = 10.0  # length of time series (seconds)
nsamples = 1000
noise_amplitude = 0.0
antenna_distance = 1

phase_shift = 0.2 * pi  # in radians

# construct time array
t = np.linspace(0.0, tmax, nsamples, endpoint=False)

# Signal A is a square wave (plus some noise)
signal = np.cos(t) + noise_amplitude * random.normal(size=(nsamples,))

# Signal B is a phase-shifted saw wave with the same period
signal_shifted = np.cos((t+phase_shift)) + noise_amplitude * random.normal(size=(nsamples,))

# calculate cross correlation of the two signals
xcorr = correlate(signal, signal_shifted)

# The peak of the cross-correlation gives the shift between the two signals
# The xcorr array goes from -nsamples to nsamples
dt = np.linspace(-t[-1], t[-1], 2 * nsamples - 1)
recovered_time_shift = dt[xcorr.argmax()]

# force the phase shift to be in [-pi:pi]
recovered_phase_shift = 2 * pi * (((0.5 + recovered_time_shift / period) % 1.0) - 0.5)

relative_error = (recovered_phase_shift - phase_shift) / (2 * pi)

print("Original phase shift: %.2f pi" % (phase_shift / pi))
print("Recovered phase shift: %.2f pi" % (recovered_phase_shift / pi))
print("Relative error: %.4f" % relative_error)

# OUTPUT:
# Original phase shift: 0.25 pi
# Recovered phase shift: 0.24 pi
# Relative error: -0.0050

# Now graph the signals and the cross-correlation
print(t)
print(signal)

plt.plot(t, signal + 2.5, color="red")
plt.plot(t, signal_shifted - 2.5, color="blue")
plt.legend(["Original", "Shifted"])

plt.show()

# Perform beam forming analysis here

