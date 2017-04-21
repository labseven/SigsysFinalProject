import numpy as np
import wave
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import time


SAMPLING_RATE = 192000

def sine_wave(hz, peak, sample_rate=SAMPLING_RATE, n_samples=SAMPLING_RATE):
    """ Compute n_samples of a sine wave.
    Given frequency, peak amplitude, sample rate"""
    length = sample_rate / hz                # How many samples per one wave
    omega = np.pi * 2 / length               # Portion of wave per sample
    xvalues = np.arange(int(length)) * omega # Array of x values
    onecycle = peak * np.sin(xvalues)        # One wave (sin of each x value)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)   # Repeat the wave to fill n_samples

def gaus_curve(length, sample_rate=SAMPLING_RATE):
    return signal.gaussian((length * sample_rate) / 1000, std=25*(length))


def make_match_filter(length, hz, peak=5000):
    sin1 = sine_wave(hz, peak, n_samples=int((length / 1000) * SAMPLING_RATE))
    gaus = gaus_curve(length)

    match_filter = gaus * sin1
    fig, ax = plt.subplots()
    # plt.plot(sin1)
    # plt.plot(gaus)
    plt.plot(match_filter)
    # fig.show()

    return match_filter

match_filter = make_match_filter(10, 1000)


with wave.open("out.wav", mode="rb") as signal_input:
    fig, ax = plt.subplots()
    signal_plt = signal_input.readframes(2)[:2]
    print(int.from_bytes(signal_plt, byteorder="little"))
    signal_plt = signal_input.readframes(1)
    print(signal_plt)
    signal_plt = signal_input.readframes(1)
    print(signal_plt)
    signal_plt = int.from_bytes(signal_input.readframes(1), byteorder="little")
    print(signal_plt)
    signal_plt = int.from_bytes(signal_input.readframes(1), byteorder="little")
    print(signal_plt)
    signal_plt = int.from_bytes(signal_input.readframes(1), byteorder="little")
    print(signal_plt)
    plt.plot(int(signal_plt))
    # fig.show()



while True:
    time.sleep(10)
