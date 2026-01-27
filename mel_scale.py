import numpy as np # pyright: ignore[reportMissingImports]

def enhance_speech(audio, fs=16000):
    digital = np.round(audio * 2**23) / 2**23
    spec = np.fft.rfft(digital)
    freqs = np.fft.rfftfreq(len(digital), 1/fs)
    
    mask = (freqs >= 500) & (freqs <= 5000)
    spec[mask] *= 1.5
    spec[~mask] *= 0.5
    
    noise = np.fft.rfft(digital - audio)
    threshold = np.abs(spec) * 0.3
    shaped = np.minimum(np.abs(noise), threshold)
    
    return np.fft.irfft(spec - shaped, len(digital))

t = np.linspace(0, 0.5, 8000)
speech = np.sin(2*np.pi*1000*t)
noisy = speech + np.random.normal(0, 0.1, len(speech))
enhanced = enhance_speech(noisy)

orig_snr = 10*np.log10(np.var(speech)/np.var(noisy-speech))
new_snr = 10*np.log10(np.var(speech)/np.var(enhanced-speech))
print(f"Before: {orig_snr:.1f} dB")
print(f"After:  {new_snr:.1f} dB")
print(f"Gain:   {new_snr-orig_snr:.1f} dB")