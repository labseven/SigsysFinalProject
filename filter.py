import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time
from signal_functions import *


match_filter = make_match_filter()
signal_in = import_wav("rec.wav")


# plot_waveform(match_filter, downsample=1, title="Match Filter", ax_labels=["Samples", "Magnitude"])
# plot_signal(signal_in, downsample=1)

envelope, convolution = get_envelope(signal_in[:600000])


# plot_waveform(convolution[:350000], downsample=10, title="5kHz Signal after Convolution", ax_labels=["Samples", "Magnitude"])
# plot_waveform(envelope[:35000], downsample=1, title="5kHz Signal after Convolution", ax_labels=["Samples", "Magnitude"])
# plot_signal(convolution)


interrupt_t, thresholds = find_intterupts(envelope)
data, packet = extract_data(interrupt_t)


plot_envelope_interrupts(envelope, interrupt_t, thresholds)

print("Packet:", packet, "Bits:", len(packet) + 1)
print("Data:", data)


while True:
    time.sleep(10)
