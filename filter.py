import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time
from signal_functions import *


match_filter = make_match_filter()
signal_in = import_wav("rec.wav")

# reference_cos = np.resize(signal_in[ : SAMPLES_PER_MS * CLOCK_MS], len(signal_in))

# print(signal_in[SAMPLES_PER_MS * CLOCK_MS - 100 : SAMPLES_PER_MS * CLOCK_MS + 100])

# plot_waveform(match_filter, downsample=1, title="Match Filter", ax_labels=["Samples", "Magnitude"])
# plot_signal(signal_in, downsample=1)

# plot_signal(signal_in)
convolution = scipy.signal.fftconvolve(match_filter, signal_in[:600000])
# envelope = np.abs(scipy.signal.hilbert(convolution[::10]))


plot_waveform(convolution, downsample=10, title="5kHz Signal after Convolution", ax_labels=["Samples", "Magnitude"])
# plot_signal(convolution)


while True:
    time.sleep(10)
