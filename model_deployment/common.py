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
    }


def _extract_features_from_signal(track_id: int, y: np.array, algos: Dict) -> Dict:
    """
    Extract features from audio signal using pre-initialized algorithms.
    """
    # --- Rhythm Features ---
    rhythm = algos["rhythm"](y)
    bpm = float(rhythm[0])
    beats = rhythm[3]
    tempogram, _ = np.histogram(np.diff(beats), bins=20)

    # --- Chroma Features ---
    # FrameGenerator cannot be reused as it holds state for the frame iteration
    frames = es.FrameGenerator(y, frameSize=32768, hopSize=8192, startFromZero=True)
    chromas = np.vstack([algos["chroma"](f) for f in frames])
    chroma_mean = chromas.mean(axis=0)
    chroma_std = chromas.std(axis=0)

    # --- MFCC Features ---
    frames_mfcc = es.FrameGenerator(y, frameSize=1025, hopSize=512, startFromZero=True)
    mfcc_all = np.vstack([algos["mfcc"](f)[1] for f in frames_mfcc])
    mfcc_mean = mfcc_all.mean(axis=0)
    mfcc_std = mfcc_all.std(axis=0)

    # --- Spectral Features ---
    sc = float(algos["centroid"](y))
    sr_feat = float(algos["rolloff"](y))
    flux = float(algos["flux"](y))
    zc = float(algos["zcr"](y))

    # --- RMS Features ---
    rms = algos["rms"](y)
    rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))
    rms_25, rms_50, rms_75 = [float(x) for x in np.percentile(rms, [25, 50, 75])]

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
