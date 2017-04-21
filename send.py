import numpy as np
from numpy import pi
import pygame, pygame.sndarray
import matplotlib.pyplot as plt
import time
import wave

SAMPLING_RATE = 192000

def sine_wave(hz, peak, len_ms, phase=0):
    """ Computes a discrete sine wave.
    Input:
        hz:     frequency of sine wave
        peak:   amplitude of the sine wave
        len_ms: length of the signal in ms
        phase:  phase offset

    Output: ndarray of type int16
    """
    num_samples = (len_ms / 1000) * SAMPLING_RATE
    num_samples_period = SAMPLING_RATE / hz                 # Number of samples in one period
    omega = pi * 2 / num_samples_period                  # Portion of wave per sample
    xvalues = np.arange(int(num_samples_period)) * omega    # Array of x values of each sample
    one_period = np.sin(xvalues + phase) * peak             # One period of the wave
    return np.resize(one_period, (num_samples,)).astype(np.int16) # Repeat the wave to fill num_samples


def cosine_wave(hz, peak, len_ms, phase=0):
    """ Computes a discrete cosine wave.
    Input:
        hz:     frequency of sine wave
        peak:   amplitude of the sine wave
        len_ms: length of the signal in ms
        phase:  phase offset

    Output: ndarray of type int16
    """
    num_samples = int((len_ms / 1000) * SAMPLING_RATE)
    num_samples_period = SAMPLING_RATE / hz                 # Number of samples in one period
    omega = pi * 2 / num_samples_period                  # Portion of wave per sample
    xvalues = np.arange(int(num_samples_period)) * omega    # Array of x values of each sample
    one_period = np.cos(xvalues + phase) * peak             # One period of the wave
    return np.resize(one_period, (num_samples,)).astype(np.int16) # Repeat the wave to fill num_samples


def make_bpsk_signal(data, hz=1000, clock_ms=10, peak=2000):
    """ Creates a BPSK signal.
    Input:
        data:       ndarray of input data [1,0] or [1,-1]
        hz:         carrier wave frequency
        peak:       carrier wave amplitude
        clock_ms:   length of one bit in ms

    Output: ndarray of type int16
    """
    # Convert to array if necessary
    if type(data) is type([]):
        data = np.array(data)

    # Rescale input if necessary
    if np.amin(data) == 0:
        data = ((data-.5)*2).astype(np.int16)

    # Create data stream (sample input)
    num_bits = len(data)
    num_samples_per_bit = int((clock_ms * SAMPLING_RATE) / 1000)
    data_stream = np.repeat(data, (num_samples_per_bit,)).astype(np.int16)

    # Create carrier wave
    carrier_wave = cosine_wave(hz, peak, clock_ms * num_bits)

    # Multiply them together to get phase shifted signal
    signal = carrier_wave * data_stream

    return signal


def plot_signal(signal, hz=1000, clock_ms=10):
    """ Plots a signal. Only plots 3 periods to each side of a bit.
    Input:
        signal:     ndarray to plot
        clock_ms:   length of one bit in ms
        hz:         frequency of the carrier
    """

    fig, ax = plt.subplots()

    # Number of samples on each side of each bit
    samples_to_show = int(SAMPLING_RATE / hz) * 2

    plt_signal = signal[:samples_to_show]

    # Number of bits in signal
    num_bits = int((len(signal) / SAMPLING_RATE) / (clock_ms / 1000))

    # Add samples in frame around each bit change
    for i in range(1, num_bits):
        frame_start = int(i * clock_ms * SAMPLING_RATE / 1000) - samples_to_show
        frame_end   = int(i * clock_ms * SAMPLING_RATE / 1000) + samples_to_show

        plt_signal = np.append(plt_signal, signal[frame_start:frame_end])

    # Show the plot
    plt.plot(plt_signal)
    fig.show()


def save_signal(signal, filename="out.wav"):
    """ Saves a signal to a file.
    Input:
        signal: ndarray or list to save
        filename: name of file to be saved to
    """

    with wave.open(filename, "w") as out_file:
        out_file.setnchannels(2)
        out_file.setsampwidth(2)
        out_file.setframerate(SAMPLING_RATE)

        out_file.writeframes(signal)

signal = make_bpsk_signal([1, 0, 1, 1], clock_ms=100)

plot_signal(signal, clock_ms=100)
save_signal(signal)


pygame.quit()

while True:
    time.sleep(10)
