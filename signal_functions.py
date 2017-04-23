import numpy as np
from numpy import pi
import wave
import scipy
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
import math
import pygame.sndarray
import time

SAMPLING_RATE = 192000
SAMPLES_PER_MS = int(SAMPLING_RATE / 1000)

CARRIER_FREQ = 5000
CLOCK_MS = 100

NUM_BITS_TRANSFERED = 14
NUM_BITS_DATA = 8

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

def gaus_curve(length_ms, sample_rate=SAMPLING_RATE):
    return signal.gaussian((length_ms * sample_rate) / 1000, std=25*(length_ms))


def make_match_filter(length_ms=CLOCK_MS, hz=CARRIER_FREQ, peak=1000):
    cos1 = cosine_wave(hz, peak, length_ms)
    gaus = gaus_curve(length_ms)

    match_filter = gaus * cos1
    return match_filter


def make_bpsk_signal(data, hz=CARRIER_FREQ, clock_ms=CLOCK_MS, peak=2000):
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

    # Resize data in to match each  (sample input)
    num_bits = len(data)
    num_samples_per_bit = int((clock_ms * SAMPLING_RATE) / 1000)
    data_stream = np.repeat(data, (num_samples_per_bit,)).astype(np.int16)

    # Create carrier wave
    carrier_wave = cosine_wave(hz, peak, clock_ms * num_bits)

    # Multiply them together to get phase shifted signal
    signal = carrier_wave * data_stream

    return signal


def export_wav(signal, channels=1, filename="out.wav"):
    """ Saves a signal to a file.
    Input:
        signal: ndarray or list to save
        filename: name of file to be saved to
    """

    with wave.open(filename, "w") as out_file:
        out_file.setnchannels(channels)
        out_file.setsampwidth(2)
        out_file.setframerate(SAMPLING_RATE)

        out_file.writeframes(signal)


def import_wav(filename):
    """ Imports a wave file and checks if the sampling rate matches SAMPLING_RATE

    Input:
        filename: name of wave file to import
    Return:
        numpy array of magnitudes
    """

    wave_in = scipy.io.wavfile.read(filename)
    if wave_in[0] != SAMPLING_RATE:
        raise(ValueError(   "Sampling rate of file does not match global \
                            sampling rate", wave_in[0]))
    return wave_in[1]


def plot_signal(signal, hz=CARRIER_FREQ, clock_ms=CLOCK_MS, downsample=10, title="", ax_labels=["",""]):
    """ Plots a signal. Only plots 3 periods to each side of a bit.
    Input:
        signal:     ndarray to plot
        clock_ms:   length of one bit in ms
        hz:         frequency of the carrier
    """

    signal = signal[::downsample]
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

    ax.set_title(title)
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])

    fig.show()


def plot_waveform(wave, downsample=100, title="", ax_labels=["",""]):
    fig, ax = plt.subplots()
    plt.plot(wave[::downsample])
    ax.set_title(title)
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    fig.show()

def play_wave(wave):
    pygame.mixer.pre_init(SAMPLING_RATE, -16, 1)
    pygame.init()
    sound = pygame.sndarray.make_sound(wave)
    sound.play()
    time.sleep(sound.get_length())

def get_envelope(signal_in, downsample=10):
    match_filter = make_match_filter()
    convolution = scipy.signal.fftconvolve(match_filter, signal_in)
    convolution = scipy.signal.fftconvolve(match_filter, convolution)
    return np.abs(scipy.signal.hilbert(convolution[::downsample])), convolution
