import pyaudio
import wave
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024 # Buffer size

print("Recording")
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []

for i in range(0, int(RATE / CHUNK)*2):
    data = stream.read(CHUNK)
    frames.append(data)


# Stop the stream
stream.stop_stream()
stream.close()
audio.terminate()

print("Saving")

waveFile = wave.open("rec.wav", 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

print("Done")
