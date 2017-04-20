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
