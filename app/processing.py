"""Audio processing with NVIDIA Riva integration and CUDA onset detection.

Combines Riva for speech/audio models with CUDA-accelerated onset detection
for real-time performance.
"""
from typing import Dict, Optional
import logging
from .riva_integration import get_riva_analyzer
from .cuda_audio import CUDAOnsetDetector
import soundfile as sf

logger = logging.getLogger(__name__)
riva_analyzer = None
onset_detector = None


def initialize_riva():
    """Initialize Riva analyzer on startup."""
    global riva_analyzer
    riva_analyzer = get_riva_analyzer()


def initialize_onset_detector():
    """Initialize CUDA onset detector on startup."""
    global onset_detector
    onset_detector = CUDAOnsetDetector(sr=16000, fft_size=2048, hop_length=512)
    logger.info("Onset detector initialized (CUDA if available)")


def analyze_audio(path: str) -> Dict:
    """Analyze an audio file with Riva and CUDA onset detection.

    Args:
        path: filesystem path to an audio file

    Returns:
        dict containing onsets, sustain, dynamics, recommendation, and speech recognition
    """
    if riva_analyzer is None:
        logger.warning("Riva unavailable; using fallback analysis")
        return _fallback_analysis()
    
    if onset_detector is None:
        logger.warning("Onset detector not initialized")
        return _fallback_analysis()
    
    try:
        # Load audio
        audio_data, sr = sf.read(path)
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Resample if needed (Riva expects 16kHz)
        if sr != 16000:
            from scipy.signal import resample
            audio_data = resample(audio_data, int(len(audio_data) * 16000 / sr))
            sr = 16000
        
        # CUDA-accelerated onset detection
        onsets = onset_detector.detect_onsets(audio_data, threshold=2.0)
        
        # Apply bandpass filter for cleaner analysis
        filtered_audio = onset_detector.filter_audio(audio_data, lowcut=80, highcut=8000)
        
        # Extract audio features using Riva
        features = riva_analyzer.extract_audio_features(path)
        
        # Recognize speech (player commands/feedback)
        speech_data = riva_analyzer.recognize_speech(path)
        transcript = speech_data.get("transcript", "")
        
        # Generate tone recommendation
        recommendation = riva_analyzer.generate_recommendation(features, transcript)
        
        return {
            "onsets": onsets,
            "sustain": [],  # TODO: implement sustain detection
            "dynamics": {
                "rms": features.get("rms"),
                "peak": features.get("peak")
            },
            "recommendation": recommendation,
            "speech": transcript,
            "notes": "CUDA onset detection + Riva analysis",
            "processing": "GPU-accelerated (cuSignal FFT, spectral flux)"
        }
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        return _fallback_analysis()


def _fallback_analysis() -> Dict:
    """Fallback analysis when components unavailable."""
    return {
        "onsets": [],
        "sustain": [],
        "dynamics": {"rms": None, "peak": None},
        "recommendation": "Analysis unavailable; Riva or CUDA components not initialized",
        "speech": "",
        "notes": "Fallback mode"
    }
