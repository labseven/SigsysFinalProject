import numpy as np
import pygame, pygame.sndarray
import matplotlib.pyplot as plt
import time
import wave

global_sample_rate = 192000


def play_tone(in_wave, in_wave2=None):
    """Play the given NumPy array, as a sound"""
    if in_wave2 is not None:
        in_wave = np.array(zip(in_wave, in_wave2))
    else:
        in_wave = np.array(zip(in_wave, in_wave))

    sound = pygame.sndarray.make_sound(in_wave)
    sound.play(0)
    pygame.time.delay(int(sound.get_length()*1000))
    sound.stop()

def sine_wave(hz, peak, sample_rate=global_sample_rate, n_samples=global_sample_rate):
    """ Compute n_samples of a sine wave.
    Given frequency, peak amplitude, sample rate"""
    length = sample_rate / hz                # How many samples per one wave
    omega = np.pi * 2 / length               # Portion of wave per sample
    xvalues = np.arange(int(length)) * omega # Array of x values
    onecycle = peak * np.sin(xvalues)        # One wave (sin of each x value)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)   # Repeat the wave to fill n_samples

def cosine_wave(hz, peak, phase=0, sample_rate=global_sample_rate, n_samples=global_sample_rate):
    """ Compute n_samples of a cosine wave.
    Given frequency, peak amplitude, and phase shift"""
    length = sample_rate / hz                   # How many samples per one wave
    # num_cycles = (n_samples/global_sample_rate) * hz # How many cycles
    omega = np.pi * 2 / length                  # Portion of wave per sample
    xvalues = (np.arange(int(length)) * omega) + phase # Array of x values
    onecycle = peak * np.cos(xvalues)           # One wave (sin of each x value)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)   # Repeat the wave to fill n_samples


def make_bpsk_signal(data, hz=1000, clock=1, peak=2000, sample_rate=global_sample_rate):
    # Length of each bit
    num_samples = int((sample_rate)*clock)
    cos1 = cosine_wave(hz, peak, phase=0,     sample_rate=sample_rate, n_samples=num_samples)
    cos2 = cosine_wave(hz, peak, phase=np.pi, sample_rate=sample_rate, n_samples=num_samples)


    # Init with three bits for calibration
    signal = cos1
    signal = np.append(signal, cos1)
    signal = np.append(signal, cos1)
    signal = []

    print("Transmitting:", data)

    for i in data:
        if i == "1":
            signal = np.append(signal, cos1)
        elif i == "0":
            signal = np.append(signal, cos2)
        else:
            raise ValueError("data must be 0 or 1. given:", i)

    ref_tone = np.resize(cos1, (num_samples * (len(data) + 3),)).astype(np.int16)
    return signal.astype(np.int16), ref_tone


def str_to_binary(string):
    out = ""
    for c in string:
        i = bin(ord(c))
        i = '0'*(10-len(i)) + str(i)[2:]
        out += i
    return out

def plot_tone(tone, tone_ref, clock_ms=1, hz=1000):
    fig, ax = plt.subplots()

    # Number of periods must end in x.5 to make
    samples_to_show = int(global_sample_rate/hz * 3)
    plt_tone = tone[:samples_to_show]
    plt_tone_ref = tone_ref[:samples_to_show]

    print("Plotting...")
    print("Tone len:", len(tone), "Samples to show:", samples_to_show, "Number of bits:", int((len(tone)/clock_ms)/global_sample_rate))

    # Cuts out same data
    for i in range(1, int((len(tone)/clock_ms)/global_sample_rate)):
        print(tone[(clock_ms * i * global_sample_rate) - samples_to_show])
        print(tone[(clock_ms * i * global_sample_rate) + samples_to_show])
        plt_tone = np.append(plt_tone, tone[(clock_ms * i * global_sample_rate) - samples_to_show:(clock_ms * i * global_sample_rate) + samples_to_show])
        plt_tone_ref = np.append(plt_tone_ref, tone_ref[(clock_ms * i * global_sample_rate) - samples_to_show:(clock_ms * i * global_sample_rate) + samples_to_show])
    plt.plot(plt_tone)
    plt.plot(plt_tone_ref)
    fig.show()

def save_wave(tone, filename="out.wav"):
    with wave.open(filename, "w") as out_file:
        out_file.setnchannels(2)
        out_file.setsampwidth(2)
        out_file.setframerate(global_sample_rate)

        out_file.writeframes(tone)

# Pygame init
pygame.mixer.pre_init(global_sample_rate, -16, 2)
pygame.init()

tone, tone_ref = make_bpsk_signal("10010001", peak=10000, hz=1000, clock=10/1000)



# Playing and plotting tone
# plot_tone(tone, tone_ref, hz=1, clock_ms=1)
# play_tone(tone, tone_ref)
save_wave(tone)
print("Done")

# Pygame exit
pygame.quit()

while True:
    time.sleep(10)
