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
import pyaudio


SAMPLING_RATE = 192000
SAMPLES_PER_MS = int(SAMPLING_RATE / 1000)

CARRIER_FREQ = 5000
CLOCK_MS = 100

NUM_BITS_TRANSFERED = 14
NUM_BITS_DATA = 8

AUDIO_FORMAT = pyaudio.paInt16
REC_CHUNK = 1024
FRAME_REC_MS = 500

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
    omega = pi * 2 / num_samples_period                     # Portion of wave per sample
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
    """ Makes a plot of anything you give it. It does not care.
    """
    fig, ax = plt.subplots()
    plt.plot(wave[::downsample])
    ax.set_title(title)
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    fig.show()

def play_wave(wave):
    """ Plays a wave. WARNING: Blocks the program until the sound ends.
    Input: array of magnitudes
    Output: Plays sound on speakers
    """
    pygame.mixer.pre_init(SAMPLING_RATE, -16, 1)
    pygame.init()
    sound = pygame.sndarray.make_sound(wave)
    sound.play()
    time.sleep(sound.get_length())

def get_envelope(signal_in, downsample=10):
    """ Convolves a signal with a match filter twice, and then returns the envelope.
    Input:
        signal_in: The signal to process
        downsample: One in how many inputs to use for the hilbert envelope creation
    Return:
        envelope of the convolved signal
        just the convolved signal
    """
    match_filter = make_match_filter()
    # Convolving twice gets a much cleaner signal
    # needed to not trigger the schmitt trigger multiple times
    convolution = scipy.signal.fftconvolve(match_filter, signal_in)
    convolution = scipy.signal.fftconvolve(match_filter, convolution)

    return np.abs(scipy.signal.hilbert(convolution[::downsample])), convolution

def find_intterupts(envelope, high_theshold_ratio=.5, low_threshold_ratio=.35):
    """ Returns a list of times when the signal goes high using a software schmitt trigger.
    Input:
        evelope: the envelope of the signal to process
        high_theshold_ratio: ratio of the max of the signal to trigger a high_theshold
        low_threshold_ratio: ratio of the max of the signal to trigger a low_threshold
    Output:
        interrupt_t: list of times when the signal goes high
        thresholds: tuple of the high and low sthresholds
    """
    # Set thresholds based on max of the signal
    high_theshold = max(envelope) * high_theshold_ratio
    low_threshold = max(envelope) * low_threshold_ratio

    flag = False
    interrupt_t = []

    # Loop through the signal and detect rising an falling edges.
    # Records the times of rising edges. Similar to a schmitt trigger
    for x in range(len(envelope)):
        if envelope[x] < low_threshold and flag:
            flag = False
        elif envelope[x] > high_theshold and not flag:
            interrupt_t.append(x)
            flag = True

    return interrupt_t, (high_theshold, low_threshold)

def plot_envelope_interrupts(envelope, interrupt_t, thresholds):
    """ Creates a pretty plot of the signal, interrups, and thresholds.
    """
    fig, ax = plt.subplots()
    # The signalFORMAT = pyaudio.paInt16
    plt.plot(envelope)
    # Points at each interrupt
    plt.scatter(interrupt_t, np.array([1000]*len(interrupt_t)), c="r")
    # Line for low threshold and high threshold
    plt.plot([thresholds[0]]*len(envelope), c="m")
    plt.plot([thresholds[1]]*len(envelope), c="m")

    ax.set_title("Interpretting Interrupts")
    ax.set_xlabel("Samples")
    fig.show()


def extract_data(interrupt_t, clock_tolerance=.4):
    """ Interprets data from a list of times of interrupts. Returns an int of
    the data.
    Input:
        interrupt_t: list of interrupts
        clock_tolerance: tolerance of the clock (as ratio of one tick)
    Returns:
        data: int of the data
        packet: original packet
    """
    # Average number of samples per bit.
    # This has a low tolerance from the rising edge detection
    clock = int((interrupt_t[-1] - interrupt_t[0]) / (NUM_BITS_TRANSFERED - 1)) + 1

    interrupt_index = 0
    flag = 1
    packet = []

    # Record how many clock ticks there are between each interrupt
    for i in range(len(interrupt_t) - 1):
        # Length of the signal
        delta_t = interrupt_t[i + 1] - interrupt_t[i]
        # delta_t divided by the clock
        # clock_tolerance adds a tolerance to the interrupts
        for i in range(int(((delta_t + (clock * clock_tolerance)) / clock))):
            packet.append(flag)
        # Flip the flag each time
        flag = [1,0][flag]

    try:
        data = int(''.join(str(e) for e in packet[4:12]), 2)
    except ValueError:
        data = 0

    return data, packet

def record_chunk(stream):
    return np.fromstring(stream.read(REC_CHUNK), 'int16')

def check_packet(data, packet):
    # Opening
    if packet[:4] != [1, 0, 1, 0]:
        # print(packet[:3])
        return False
    # Footer
    if packet[-1] != 1:
        # print(packet[-1])
        return False
    # Ascii
    if data > 128:
        # print(data)
        return False

    return True
