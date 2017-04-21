import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time
from signal_functions import *


match_filter = make_match_filter(1000)
signal_in = import_wav("out.wav")



# plot_waveform(match_filter)
# plot_signal(signal_in, clock_ms=1000)

convolution = scipy.signal.fftconvolve(match_filter, signal_in[:SAMPLES_PER_MS])
plot_waveform(convolution)

while True:
    time.sleep(10)
