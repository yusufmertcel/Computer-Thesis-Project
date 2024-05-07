# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:53:59 2023

@author: sceli
"""

import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt

def low_passFilter(df, segment_name):
    sig_fft, X, sr, n = to_FFT(df, segment_name)
    filtered_low, filtered_high = to_IFFT(sig_fft, X, sr, n)
    return [float(abs(x)) for x in filtered_low], [float(abs(x)) for x in filtered_low] 

def to_FFT(df, segment_name):
    X = df["hiz"].values
    sig_fft = fft(X)
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    sr = 1 / (60*5)
    T = N/sr
    freq = n/T 
    
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    plt.figure(figsize = (12, 6))
    plt.plot(f_oneside, np.abs(sig_fft[:n_oneside]), 'b')
    plt.title(f"{segment_name}")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()
    
    

    t_h = 1/f_oneside / (60 * 5)

    plt.figure(figsize=(12,6))
    plt.plot(t_h, np.abs(sig_fft[:n_oneside])/n_oneside)
    #plt.xticks([12, 24, 84, 168])
    plt.xlim(0, 144)
    plt.xlabel('Period ($minute$)')
    plt.show()
    
    return sig_fft, X, sr, n

def to_IFFT(sig_fft, X, sr, n):
    sig_fft_filtered_low = sig_fft.copy()
    sig_fft_filtered_high = sig_fft.copy()
    freq = fftfreq(len(X), d=1./sr)

    # define the cut-off frequency
    cut_off = 0.0002 # 0.0002 0.00006

    # low-pass filter by assign zeros to the 
    # FFT amplitudes where the absolute 
    # frequencies smaller than the cut-off 
    sig_fft_filtered_low[np.abs(freq) > cut_off] = 0
    # high-pass filter
    sig_fft_filtered_high[np.abs(freq) < cut_off] = 0
    # get the filtered signal in time domain
    filtered_low = ifft(sig_fft_filtered_low)
    filtered_high = ifft(sig_fft_filtered_high)
    #------------ Filtered plot
    #plot the filtered signal
    plt.figure(figsize = (12, 6))
    plt.plot(n, X, label="Non-filtered")
    plt.plot(n, filtered_low.real, label="Filtered")
    plt.title("Low Filtered Data")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.plot(n, filtered_high.real)
    plt.title("High Filtered Data")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
    # plot the FFT amplitude before and after
    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.stem(freq, np.abs(sig_fft), 'b', \
              markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    #plt.xlim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.subplot(122)
    plt.stem(freq, np.abs(sig_fft_filtered_low), 'b', \
              markerfmt=" ", basefmt="-b")
    plt.title('After filtering-low')
    #plt.xlim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.stem(freq, np.abs(sig_fft), 'b', \
              markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    #plt.xlim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.subplot(122)
    plt.stem(freq, np.abs(sig_fft_filtered_high), 'b', \
              markerfmt=" ", basefmt="-b")
    plt.title('After filtering-high')
    #plt.xlim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()
    
    
    return filtered_low, filtered_high




    
    
    
    
    

