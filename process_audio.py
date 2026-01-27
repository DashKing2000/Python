
import numpy as np
from scipy.io import wavfile

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

# Load your audio file
fs, audio = wavfile.read('noisy_speech.wav')

# Convert to float
audio = audio.astype(float) / np.max(np.abs(audio))

# Enhance it
enhanced = enhance_speech(audio, fs)

# Save result
enhanced_int = np.int16(enhanced * 32767)
wavfile.write('enhanced_speech.wav', fs, enhanced_int)

print("Enhanced audio saved as 'enhanced_speech.wav'")