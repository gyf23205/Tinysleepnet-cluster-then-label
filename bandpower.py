def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper
    import numpy as np

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import numpy as np

# def bandpower(pxx, freqs, band_freq, total_freq_range=None, relative=False, ravel_if_one_band=True):
#     """Compute band power from power spectrogram.
#
#     Arguments:
#     pxx -- power spectrogram from from multitaper_spectrogram, size=(window_num, freq_point_num1, channel_num), or a list of them for each band
#     freqs -- in Hz, size=(nfft//2+1,), or a list of them for each band
#     band_freq -- bands to compute, [[band1_start,band1_end],[band2_start,band2_end],...] in Hz
#
#     Keyword arguments:
#     total_freq_range -- default None, total range of frequency in a two-element list in Hz, if None and relative is True, use the maximum range in band_freq
#     relative -- default False, whether to compute relative band power w.r.t the total frequency range
#     ravel_if_one_band -- default True, whether to only return the first element if one band
#
#     Outputs:
#     band power, size=(window_num, channel_num) or a list of them for each band
#     indices in freqs for each band
#     """
#     if not hasattr(band_freq[0],'__iter__'):
#         band_freq = [band_freq]
#     band_num = len(band_freq)
#
#     if relative and total_freq_range is None:
#         total_freq_range = [min(min(bf) for bf in band_freq),max(max(bf) for bf in band_freq)]
#     if relative:
#         total_findex = np.where(np.logical_and(freqs>=total_freq_range[0], freqs<total_freq_range[1]))[0]
#
#     bp = []
#     band_findex = []
#     for bi in range(band_num):
#         band_findex.append(np.where(np.logical_and(freqs>=band_freq[bi][0], freqs<band_freq[bi][1]))[0])
#
#         if relative:
#             bp.append(pxx[:,band_findex[-1],:].sum(axis=1)*1.0/pxx[:,total_findex,:].sum(axis=1))
#         else:
#             bp.append(pxx[:,band_findex[-1],:].sum(axis=1)*(freqs[1]-freqs[0]))
#
#     if ravel_if_one_band and band_num==1:
#         bp = bp[0]
#         band_findex = band_findex[0]
#
#     return bp, band_findex
#
#
# if __name__=='__main__':
#     import pdb
#     from scipy import io as sio
#     from multitaper_spectrogram import *
#
#     ff = sio.loadmat(r'C:\Users\BMW_HMI\Desktop\ExampleCodeForSpectrograms\multitaper_example.mat')
#     EEG = ff['eeg']
#     EEG_after_detrend = ff['eeg_after_detrend']
#     ss = ff['ss']
#
#     Fs = 200
#     NW = 2
#     band_freq = [0.5,55]
#     #band_freq = [[0.5,25],[25,55]]
#     window_length = 2
#     window_step = 0.2
#
#     pdb.set_trace()
#     mt_pxx, freqs = multitaper_spectrogram(EEG, Fs, NW, window_length, window_step)
#     bp = bandpower(mt_pxx, freqs, band_freq, total_freq_range=None, relative=False)
#     print(bp)
