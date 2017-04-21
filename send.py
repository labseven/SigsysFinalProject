from signal_functions import *
import time


signal = make_bpsk_signal([1, 0, 1, 1])

# plot_signal(signal)
export_wav(signal)

# 
# while True:
#     time.sleep(10)
