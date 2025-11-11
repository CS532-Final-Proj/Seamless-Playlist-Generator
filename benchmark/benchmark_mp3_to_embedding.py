import time
import numpy as np
import librosa
import soundfile as sf
import essentia.standard as es
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import sys
import os

# Load pre-trained scaler and PCA
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def standardize_audio(mp3_path, out_path):
    """Standardize MP3 to mono WAV at 22,050 Hz, middle 30 seconds."""
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)
    mid = len(y) // 2
    crop = y[mid - 15 * sr: mid + 15 * sr]
    sf.write(out_path, crop, sr, subtype="FLOAT")
    return out_path

def extract_features(wav_path):
    """Extract audio features using Essentia."""
    y = es.MonoLoader(filename=wav_path, sampleRate=22050)()

    # Rhythm
    rhythm = es.RhythmExtractor2013()(y)
    bpm = rhythm[0]
    beats = rhythm[3]
    tempogram, _ = np.histogram(np.diff(beats), bins=20)

    # Chroma
    frames = es.FrameGenerator(y, frameSize=32768, hopSize=8192, startFromZero=True)
    chromas = np.vstack([es.Chromagram()(f) for f in frames])
    chroma_mean = chromas.mean(axis=0)
    chroma_std = chromas.std(axis=0)

    # MFCC
    frames_mfcc = es.FrameGenerator(y, frameSize=1025, hopSize=512, startFromZero=True)
    mfcc_all = np.vstack([es.MFCC()(f)[1] for f in frames_mfcc])
    mfcc_mean = mfcc_all.mean(axis=0)
    mfcc_std = mfcc_all.std(axis=0)

    # Spectral
    sc = es.Centroid()(y)
    sr = es.RollOff()(y)
    flux = es.Flux()(y)
    zc = es.ZeroCrossingRate()(y)

    # RMS
    rms = es.RMS()(y)
    rms_mean, rms_std = np.mean(rms), np.std(rms)
    rms_25, rms_50, rms_75 = np.percentile(rms, [25, 50, 75])

    # Combine features
    feature_row = {"bpm": bpm}
    feature_row.update({f"tempogram_{i}": v for i, v in enumerate(tempogram)})
    feature_row.update({f"chroma_mean_{i}": v for i, v in enumerate(chroma_mean)})
    feature_row.update({f"chroma_std_{i}": v for i, v in enumerate(chroma_std)})
    feature_row.update({f"mfcc_mean_{i}": v for i, v in enumerate(mfcc_mean)})
    feature_row.update({f"mfcc_std_{i}": v for i, v in enumerate(mfcc_std)})
    feature_row.update({
        "spectral_centroid": sc, "spectral_rolloff": sr,
        "spectral_flux": flux, "zero_crossing": zc,
        "rms_mean": rms_mean, "rms_std": rms_std,
        "rms_25": rms_25, "rms_50": rms_50, "rms_75": rms_75
    })

    return feature_row

def get_embedding(mp3_path):
    """Process MP3 to embedding vector."""
    temp_wav = "temp_benchmark.wav"
    standardize_audio(mp3_path, temp_wav)
    features = extract_features(temp_wav)
    X = np.array(list(features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    emb = pca.transform(X_scaled)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # Clean up temp file
    os.remove(temp_wav)
    return emb_norm[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_mp3_to_embedding.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        sys.exit(1)

    mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
    if not mp3_files:
        print(f"No MP3 files found in '{folder_path}'.")
        sys.exit(1)

    times = []
    for mp3_file in mp3_files:
        mp3_path = os.path.join(folder_path, mp3_file)
        try:
            start = time.time()
            embedding = get_embedding(mp3_path)
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            print(f"{mp3_file}: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")
            continue

    if not times:
        print("No files processed successfully.")
        sys.exit(1)

    import statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    print("\nBenchmark Results:")
    print(f"Number of MP3s processed: {len(times)}")
    print(f"Mean time per MP3: {mean_time:.2f} seconds")
    print(f"Median time per MP3: {median_time:.2f} seconds")
    print(f"Std Dev: {std_dev:.2f} seconds")
    print(f"Min time: {min_time:.2f} seconds")
    print(f"Max time: {max_time:.2f} seconds")
    print(f"Total time: {sum(times):.2f} seconds")
    print(f"Embeddings per second: {len(times) / sum(times):.2f}")
    print(f"Final embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")