from signal_functions import *
import time
import sys


for c in sys.argv[1]:
    character_bin = bin(ord(c))[2:]
    character_bin_pad = [0 for i in range(8-len(character_bin))] + [[0,1][int(a)] for a in character_bin]

    data = [1, 0, 1, 0] + character_bin_pad + [1, 0]

    # data = [1, 0, 0, 1]
    print("Making signal. Hz:", CARRIER_FREQ, "\tClock:", str(CLOCK_MS) + "ms", \
            "\tSampling Rate:", SAMPLING_RATE)

    print("Data:", data)

    signal = make_bpsk_signal(data)


    export_wav(signal)


    # plot_waveform(signal, downsample=1, title="Signal of [1,0,0,1]", ax_labels=["Samples", "Magnitude"])

    # while True:
    #     time.sleep(10)

    play_wave(signal)
    time.sleep(NUM_BITS_TRANSFERED * CLOCK_MS / 2000)

while True:
    time.sleep(1)
