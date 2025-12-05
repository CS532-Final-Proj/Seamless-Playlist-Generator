from typing import Dict
import essentia.standard as es
import numpy as np
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)


def get_essentia_algos():
    """Initialize reusable Essentia algorithms."""
    return {
        "rhythm": es.RhythmExtractor2013(),
        "chroma": es.Chromagram(),
        "mfcc": es.MFCC(),
        "centroid": es.Centroid(),
        "rolloff": es.RollOff(),
        "flux": es.Flux(),
        "zcr": es.ZeroCrossingRate(),
        "rms": es.RMS(),
        "window": es.Windowing(type="hann"),
        "spectrum": es.Spectrum(),
    }


def _extract_features_from_signal(track_id: int, y: np.array, algos: Dict) -> Dict:
    """
    Extract features from audio signal using pre-initialized algorithms.
    """
    # --- Rhythm Features ---
    # RhythmExtractor2013 takes the full signal
    rhythm = algos["rhythm"](y)
    bpm = float(rhythm[0])
    beats = rhythm[3]
    tempogram, _ = np.histogram(np.diff(beats), bins=20)

    # --- Spectral, Time-domain & MFCC Features (Frame-based) ---
    # We compute these on frames to get statistics (mean/std)

    FRAME_SIZE = 2048
    HOP_SIZE = 1024

    # Reuse stateless algos
    centroid_algo = algos["centroid"]
    rolloff_algo = algos["rolloff"]
    zcr_algo = algos["zcr"]
    rms_algo = algos["rms"]
    window_algo = algos["window"]
    spectrum_algo = algos["spectrum"]
    mfcc_algo = algos["mfcc"]

    # Flux is stateful (diff between frames), so we instantiate it locally
    flux_algo = es.Flux()

    # Generators
    frames = es.FrameGenerator(
        y, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True
    )

    centroids = []
    rolloffs = []
    fluxes = []
    zcrs = []
    rmses = []
    mfcc_list = []

    for frame in frames:
        # Time-domain features
        zcrs.append(zcr_algo(frame))
        rmses.append(rms_algo(frame))

        # Spectral features
        # Windowing and Spectrum adapt to input size (FRAME_SIZE)
        spec = spectrum_algo(window_algo(frame))

        centroids.append(centroid_algo(spec))
        rolloffs.append(rolloff_algo(spec))
        fluxes.append(flux_algo(spec))

        # MFCC (uses same spec)
        _, mfcc_coeffs = mfcc_algo(spec)
        mfcc_list.append(mfcc_coeffs)

    # Aggregate Spectral/Time
    sc = float(np.mean(centroids)) if centroids else 0.0
    sr_feat = float(np.mean(rolloffs)) if rolloffs else 0.0
    flux = float(np.mean(fluxes)) if fluxes else 0.0
    zc = float(np.mean(zcrs)) if zcrs else 0.0

    # RMS stats
    if rmses:
        rms_arr = np.array(rmses)
        rms_mean = float(np.mean(rms_arr))
        rms_std = float(np.std(rms_arr))
        rms_25, rms_50, rms_75 = [
            float(x) for x in np.percentile(rms_arr, [25, 50, 75])
        ]
    else:
        rms_mean, rms_std, rms_25, rms_50, rms_75 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Aggregate MFCC
    if mfcc_list:
        mfcc_all = np.vstack(mfcc_list)
        mfcc_mean = mfcc_all.mean(axis=0)
        mfcc_std = mfcc_all.std(axis=0)
    else:
        mfcc_mean = np.zeros(13)
        mfcc_std = np.zeros(13)

    # --- Chroma Features ---
    # Chromagram takes time-domain frames with different size (32768)
    # FrameGenerator cannot be reused as it holds state for the frame iteration
    frames_chroma = es.FrameGenerator(
        y, frameSize=32768, hopSize=8192, startFromZero=True
    )
    chromas = np.vstack([algos["chroma"](f) for f in frames_chroma])
    chroma_mean = chromas.mean(axis=0)
    chroma_std = chromas.std(axis=0)

    # Combine
    features = [bpm, sc, sr_feat, flux, zc, rms_mean, rms_std, rms_25, rms_50, rms_75]
    features.extend(tempogram.astype(float).tolist())
    features.extend(chroma_mean.astype(float).tolist())
    features.extend(chroma_std.astype(float).tolist())
    features.extend(mfcc_mean.astype(float).tolist())
    features.extend(mfcc_std.astype(float).tolist())

    return {"track_id": int(track_id), "features": features}


# Define schema for feature extraction output
feature_schema = StructType(
    [
        StructField("track_id", IntegerType(), False),
        StructField("features", ArrayType(FloatType()), False),
    ]
)

# Global storage for worker process
_worker_algos = None


def init_worker():
    """Initialize Essentia algorithms in the worker process."""
    global _worker_algos
    _worker_algos = get_essentia_algos()


def safe_extract_worker(track_id, mp3_path):
    """
    Worker function to load and extract features.
    Runs in a separate process to isolate segfaults.
    """
    import essentia.standard as es
    import numpy as np

    try:
        # Load
        loader = es.MonoLoader(filename=mp3_path, sampleRate=22050)
        y = loader()

        # Trim
        # sr = 22050
        # duration_samples = len(y)
        # target_samples = 30 * sr
        # if duration_samples > target_samples:
        #     mid = duration_samples // 2
        #     start = mid - (target_samples // 2)
        #     end = start + target_samples
        #     y = y[start:end]

        # Extract
        # Use the global algos initialized in this process
        feat_result = _extract_features_from_signal(track_id, y, _worker_algos)

        # Convert numpy arrays to python lists for PySpark serialization
        if feat_result and "features" in feat_result:
            features = feat_result["features"]
            if isinstance(features, np.ndarray):
                feat_result["features"] = features.tolist()

        return feat_result
    except Exception as e:
        raise e
