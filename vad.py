from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import matplotlib.pyplot as plt
import numpy as np

model = load_silero_vad()
SR = 16000

# Read and convert
wav = read_audio("audio_samples/audio_clean.wav", SR)
wav = wav.numpy()  # Convert PyTorch tensor to NumPy array
wav = wav / (np.max(np.abs(wav)) + 1e-6)

# Plot waveform
time = np.linspace(0, len(wav) / SR, num=len(wav))
plt.figure(figsize=(14, 4))
plt.plot(time, wav)
plt.title("Waveform of audio_clean.wav")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("waveform_normalised_plot.png")

# Get VAD timestamps
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True
)

print("🧠 Detected speech timestamps:", speech_timestamps)
