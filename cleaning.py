import os
import numpy as np
import librosa
import subprocess
import librosa
import parselmouth
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def convert_to_wav(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_path, '-y'
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False

def extract_voice_segment(file_path):
    model = load_silero_vad()
    wav = read_audio(file_path, sampling_rate=16000)

    if wav.ndim > 1:
        wav = wav[0]

    speech_ts = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.1)

    if not speech_ts:
        print("no voice")
        return np.array([]), 16000
    print(speech_ts)
    segments = []
    for t in speech_ts:
        start = int(t['start'] * 16000)
        end = int(t['end'] * 16000)
        if start < end:
            segments.append(wav[start:end])

    if not segments:
        return np.array([]), 16000

    voice_only = np.concatenate(segments)
    return voice_only, 16000


def compute_snr(y):
    rms_signal = np.sqrt(np.mean(y ** 2))
    rms_noise = np.std(y - np.mean(y))
    snr = 20 * np.log10((rms_signal + 1e-6) / (rms_noise + 1e-6))
    return snr

def compute_hnr(y, sr):
    snd = parselmouth.Sound(y, sr)
    harmonicity = snd.to_harmonicity_ac(time_step=0.01, minimum_pitch=75)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    return hnr



def compute_spectral_tone(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(
        y=y, n_fft=2048, hop_length=512))
    return 1 - flatness  # higher = more tonal

def analyze_clarity(file_path):

    full, _ = librosa.load(file_path, sr=16000, mono=True)
    voice_audio, sr = extract_voice_segment(file_path)

    if len(voice_audio) == 0:
        print("no speech")
        return

    # normalize voice
    voice_audio = voice_audio / (np.max(np.abs(voice_audio)) + 1e-6)

    snr = compute_snr(voice_audio)
    hnr = compute_hnr(voice_audio, sr)

    tone = compute_spectral_tone(voice_audio, sr)
    voiced_ratio = len(voice_audio) / len(full) 

    snr_norm = np.clip(snr / 50, 0, 1)
    hnr_norm = np.clip(hnr / 50, 0, 1)

    clarity = (
        snr_norm * 0.35 + (hnr_norm*0.25) + (tone*0.25) + (voiced_ratio*0.15)
    )

    clarity_score = round(clarity * 100, 2)
    action = "clear" if clarity_score >= 30 else "reupload"

    print(f"Clarity Score: {clarity_score}%")
    print(f"SNR norm:       {snr_norm:.2f}")
    print(f"HNR norm:       {hnr_norm:.2f}")
    print(f"tonal content:  {tone:.2f}")
    print(f"voiced ratio:   {voiced_ratio:.2f}")
    print("action:", action)

if __name__ == "__main__":
    audio_file = "audio_samples/audio_noisy2.mp3"
    analyze_clarity(audio_file)
