def enhance_speech(audio, fs=16000):
    digital = np.round(audio * 2**23) / 2**23  # PCM1808
    spec = np.fft.rfft(digital)
    freqs = np.fft.rfftfreq(len(digital), 1/fs)
    
    mask = (freqs >= 500) & (freqs <= 5000)
    spec[mask] *= 1.5
    spec[~mask] *= 0.5
    
    noise = np.fft.rfft(digital - audio)
    threshold = np.abs(spec) * 0.3
    shaped = np.minimum(np.abs(noise), threshold)
    
    return np.fft.irfft(spec - shaped, len(digital))