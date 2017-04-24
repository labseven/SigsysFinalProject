from signal_functions import *
from collections import deque
from string import ascii_lowercase

audio = pyaudio.PyAudio()

stream = audio.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=REC_CHUNK)

rec_samples = (CLOCK_MS * NUM_BITS_TRANSFERED + 1000) * SAMPLES_PER_MS
analyze_samples = (FRAME_REC_MS * SAMPLES_PER_MS)
recording = deque(maxlen=rec_samples)

stopping = False

message = ""

while not stopping:
    i = 0

    while True:
        recording.extend(record_chunk(stream))

        if(i % 50 == 0):
            signal_in = np.array(recording)
            envelope, convolution = get_envelope(signal_in)
            # print('.' * ((i%1000)//100 + 1), " " * 10, end="\r")
            # print(max(envelope[len(envelope)//3:]), " "*10, end="\r")
            # print("#" * int(max(envelope[len(envelope)//3:])//10**11))

            if max(envelope) > (2.2 * 10**14):
                interrupt_t, thresholds = find_intterupts(envelope)
                if len(interrupt_t) > 4:
                    # print(interrupt_t[-1] - interrupt_t[0], ((interrupt_t[3] - interrupt_t[0])/4) * (NUM_BITS_TRANSFERED-1))
                    if (interrupt_t[-1] - interrupt_t[0]) > ((interrupt_t[3] - interrupt_t[0])/4) * (NUM_BITS_TRANSFERED-1):
                        for i in range(100):
                            recording.extend(record_chunk(stream))
                        break

        i+=1



    signal_in = np.array(recording)
    envelope, convolution = get_envelope(signal_in)

    interrupt_t, thresholds = find_intterupts(envelope)

    data, packet = extract_data(interrupt_t)


    plot_envelope_interrupts(envelope, interrupt_t, thresholds)

    # print("Packet:", packet, "Bits:", len(packet) + 1)
    # print("Data:", chr(data))

    if check_packet(data, packet):
        message += chr(data)
        print(message)

    else:
        print("Got garbage data:", packet)

    # print("Highest signal:", max(envelope))
    recording.clear()

    if data == ord("|"):
        print()
        print("Quitting...")
        stopping = True



stream.stop_stream()
stream.close()
audio.terminate()
