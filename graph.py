import numpy as np
import matplotlib.pyplot as plt

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

# Test
t = np.linspace(0, 0.5, 8000)
speech = np.sin(2*np.pi*1000*t)
noisy = speech + np.random.normal(0, 0.1, len(speech))
enhanced = enhance_speech(noisy)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t[:200], noisy[:200], label='Noisy', alpha=0.7)
plt.plot(t[:200], enhanced[:200], label='Enhanced', linewidth=2)
plt.title('Waveform Comparison')
plt.legend()
plt.xlabel('Time (s)')

plt.subplot(1, 2, 2)
plt.plot(t[:200], speech[:200], 'g-', label='Original', linewidth=2)
plt.plot(t[:200], enhanced[:200], 'b--', label='Enhanced', alpha=0.7)
plt.title('Enhancement Quality')
plt.legend()
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig('results.png')
print("Saved: results.png")
plt.show()