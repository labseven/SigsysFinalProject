import alsaaudio, wave, numpy


print(alsaaudio.pcms())

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, device="sysdefault:CARD=PCH")
inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(1024)

w = wave.open('test.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(44100)

while True:
    l, data = inp.read()
    # print(data)
    a = numpy.fromstring(data, dtype='int16')
    print(numpy.abs(a).mean())
    w.writeframes(data)
