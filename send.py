from signal_functions import *
import time
import sys

character_bin = bin(ord(sys.argv[1][0]))[2:]
character_bin_pad = [0 for i in range(8-len(character_bin))] + [[0,1][int(a)] for a in character_bin]

print(character_bin_pad)

data = [1, 0, 1, 0] + character_bin_pad + [1, 0]

# data = [1, 0, 0, 1]
print("Making signal. Hz:", CARRIER_FREQ, "\tClock:", str(CLOCK_MS) + "ms", \
        "\tSampling Rate:", SAMPLING_RATE)

print("Data:", data)

signal = make_bpsk_signal(data)


export_wav(signal)


# plot_signal(signal, downsample=1, title="Signal of [1,0,0,1]", ax_labels=["Samples", "Magnitude"])
# while True:
#     time.sleep(10)

play_wave(signal)
