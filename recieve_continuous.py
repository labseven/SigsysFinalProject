from signal_functions import *
from collections import deque


audio = pyaudio.PyAudio()

stream = audio.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=REC_CHUNK)

num_frames = (MAX_REC_MS//1000) * (SAMPLING_RATE // REC_CHUNK)
recording = deque(maxlen=num_frames)
i = 0
stopping = False


while True:
    recording.append(record_chunk(stream))

    print(len(recording))
    i += 1



stream.stop_stream()
stream.close()
audio.terminate()
