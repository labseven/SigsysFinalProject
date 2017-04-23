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
convolution = scipy.signal.fftconvolve(match_filter, convolution)
envelope = np.abs(scipy.signal.hilbert(convolution[::100]))


# plot_waveform(convolution, downsample=10, title="5kHz Signal after Convolution", ax_labels=["Samples", "Magnitude"])
# plot_waveform(envelope, downsample=1, title="5kHz Signal after Convolution", ax_labels=["Samples", "Magnitude"])
# plot_signal(convolution)


high_theshold = max(envelope) * .5
low_threshold = max(envelope) * .35
print(low_threshold, high_theshold)
flag = False
interrupt_t = []

for x in range(len(envelope)):
    if envelope[x] < low_threshold and flag:
        flag = False
    elif envelope[x] > high_theshold and not flag:
        interrupt_t.append(x)
        flag = True

interrupt_plt = np.vstack((interrupt_t, np.array([1000]*len(interrupt_t)))).T
print(interrupt_plt)

fig, ax = plt.subplots()
plt.plot(envelope)
plt.scatter(interrupt_t, np.array([1000]*len(interrupt_t)), c="r")
plt.plot([high_theshold]*len(envelope), c="m")
plt.plot([low_threshold]*len(envelope), c="m")
fig.show()


clock = int((interrupt_t[-1] - interrupt_t[0]) / (NUM_BITS_TRANSFERED - 1))
print(clock)

interrupt_index = 0
flag = True
packet = []

for i in range(len(interrupt_t) - 1):
    delta_t = interrupt_t[i + 1] - interrupt_t[i]
    packet.append([flag] * int(((delta_t + clock * .4) / clock)))
    flag = not flag

print(packet)
print(len(packet) + 1)

while True:
    time.sleep(10)
