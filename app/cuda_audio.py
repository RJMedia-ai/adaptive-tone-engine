"""CUDA-accelerated audio processing using cuSignal and cuPy.

Implements real-time onset detection, spectral analysis, and filtering
using GPU-accelerated FFT and signal processing primitives.
"""
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np

try:
    import cupy as cp
    from cusignal import windows, spectral_analysis
    import cusignal as signal
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = np
    signal = None

logger = logging.getLogger(__name__)


class CUDAOnsetDetector:
    """GPU-accelerated onset detection using CUDA FFT and signal processing."""
    
    def __init__(self, sr: int = 16000, fft_size: int = 2048, hop_length: int = 512):
        """
        Args:
            sr: Sample rate (Hz)
            fft_size: FFT window size
            hop_length: Samples between frames
        """
        self.sr = sr
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.cuda_available = CUDA_AVAILABLE
        
        if not self.cuda_available:
            logger.warning("CUDA not available; using CPU fallback")
    
    def detect_onsets(self, audio: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detect note onsets using GPU-accelerated spectral flux.
        
        Args:
            audio: Audio waveform (mono, numpy array)
            threshold: Detection sensitivity (std devs above mean)
            
        Returns:
            List of sample indices where onsets occur
        """
        if not self.cuda_available:
            return self._onset_detection_cpu(audio, threshold)
        
        try:
            # Transfer audio to GPU
            audio_gpu = cp.asarray(audio, dtype=cp.float32)
            
            # Compute STFT on GPU
            f, t, Sxx = self._stft_gpu(audio_gpu)
            
            # Compute spectral flux (energy onset function)
            spectral_flux = self._spectral_flux_gpu(Sxx)
            
            # Detect peaks in flux
            onsets = self._peak_detection_gpu(spectral_flux, threshold)
            
            # Convert frame indices to sample indices
            onset_samples = [int(idx * self.hop_length) for idx in onsets]
            
            logger.info(f"Detected {len(onset_samples)} onsets via CUDA")
            return onset_samples
        
        except Exception as e:
            logger.error(f"CUDA onset detection failed: {e}. Falling back to CPU.")
            return self._onset_detection_cpu(audio, threshold)
    
    def filter_audio(self, audio: np.ndarray, lowcut: float = 80, highcut: float = 8000) -> np.ndarray:
        """Apply GPU-accelerated bandpass filter.
        
        Args:
            audio: Audio waveform
            lowcut: Lower cutoff frequency (Hz)
            highcut: Upper cutoff frequency (Hz)
            
        Returns:
            Filtered audio
        """
        if not self.cuda_available:
            return self._filter_cpu(audio, lowcut, highcut)
        
        try:
            audio_gpu = cp.asarray(audio, dtype=cp.float32)
            
            # Compute filter coefficients (CPU)
            nyquist = self.sr / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = self._butter_coeffs(low, high)
            
            # Apply IIR filter on GPU
            b_gpu = cp.asarray(b, dtype=cp.float32)
            a_gpu = cp.asarray(a, dtype=cp.float32)
            filtered = cp.asnumpy(self._lfilter_gpu(b_gpu, a_gpu, audio_gpu))
            
            logger.info(f"Applied CUDA bandpass filter ({lowcut}-{highcut} Hz)")
            return filtered
        
        except Exception as e:
            logger.error(f"CUDA filtering failed: {e}. Falling back to CPU.")
            return self._filter_cpu(audio, lowcut, highcut)
    
    def compute_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram on GPU.
        
        Returns:
            (frequencies, times, spectrogram_magnitude)
        """
        if not self.cuda_available:
            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(audio, fs=self.sr, nperseg=self.fft_size)
            return f, t, np.abs(Sxx)
        
        try:
            audio_gpu = cp.asarray(audio, dtype=cp.float32)
            f, t, Sxx = self._stft_gpu(audio_gpu)
            return f, t, cp.asnumpy(cp.abs(Sxx))
        except Exception as e:
            logger.error(f"CUDA spectrogram failed: {e}")
            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(audio, fs=self.sr, nperseg=self.fft_size)
            return f, t, np.abs(Sxx)
    
    # Private GPU methods
    
    def _stft_gpu(self, audio_gpu: 'cp.ndarray') -> Tuple[np.ndarray, np.ndarray, 'cp.ndarray']:
        """Short-time Fourier Transform on GPU."""
        # Window function
        window = cp.hanning(self.fft_size, dtype=cp.float32)
        
        # Framing
        frames = cp.lib.stride_tricks.as_strided(
            audio_gpu,
            shape=(
                (audio_gpu.shape[0] - self.fft_size) // self.hop_length + 1,
                self.fft_size
            ),
            strides=(self.hop_length * audio_gpu.itemsize, audio_gpu.itemsize)
        )
        
        # Apply window and FFT
        windowed = frames * window[cp.newaxis, :]
        fft_result = cp.fft.rfft(windowed, axis=1)
        
        # Compute frequency/time axes
        freqs = np.fft.rfftfreq(self.fft_size, 1 / self.sr)
        times = np.arange(fft_result.shape[0]) * self.hop_length / self.sr
        
        return freqs, times, fft_result
    
    def _spectral_flux_gpu(self, fft_result: 'cp.ndarray') -> 'cp.ndarray':
        """Compute spectral flux (energy onset function)."""
        magnitude = cp.abs(fft_result)
        # Flux: sum of positive differences between consecutive frames
        diff = cp.diff(magnitude, axis=0, prepend=0)
        flux = cp.sum(cp.maximum(diff, 0), axis=1)
        return flux
    
    def _peak_detection_gpu(self, signal_data: 'cp.ndarray', threshold: float) -> List[int]:
        """Detect peaks in signal using threshold."""
        mean = cp.mean(signal_data)
        std = cp.std(signal_data)
        threshold_val = mean + threshold * std
        
        # Find peaks
        peaks = cp.where(signal_data > threshold_val)[0]
        peaks_list = cp.asnumpy(peaks).tolist()
        
        return peaks_list
    
    def _lfilter_gpu(self, b: 'cp.ndarray', a: 'cp.ndarray', x: 'cp.ndarray') -> 'cp.ndarray':
        """IIR filter on GPU (simple direct form II)."""
        # This is a simplified version; for production use scipy's sosfilt
        y = cp.zeros_like(x)
        for n in range(len(x)):
            y[n] = b[0] * x[n]
            for k in range(1, min(len(b), n + 1)):
                y[n] += b[k] * x[n - k]
            for k in range(1, min(len(a), n + 1)):
                y[n] -= a[k] * y[n - k]
            y[n] /= a[0]
        return y
    
    def _butter_coeffs(self, lowcut: float, highcut: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Butterworth filter coefficients (CPU)."""
        from scipy.signal import butter
        b, a = butter(order, [lowcut, highcut], btype='band')
        return b, a
    
    # CPU fallback methods
    
    def _onset_detection_cpu(self, audio: np.ndarray, threshold: float) -> List[int]:
        """CPU-based onset detection using scipy."""
        from scipy.signal import spectrogram
        
        f, t, Sxx = spectrogram(audio, fs=self.sr, nperseg=self.fft_size)
        
        # Spectral flux
        magnitude = np.abs(Sxx)
        diff = np.diff(magnitude, axis=0, prepend=0)
        flux = np.sum(np.maximum(diff, 0), axis=1)
        
        # Peak detection
        mean = np.mean(flux)
        std = np.std(flux)
        threshold_val = mean + threshold * std
        peaks = np.where(flux > threshold_val)[0]
        
        onset_samples = [int(idx * self.hop_length) for idx in peaks]
        logger.info(f"Detected {len(onset_samples)} onsets via CPU")
        return onset_samples
    
    def _filter_cpu(self, audio: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        """CPU-based bandpass filter using scipy."""
        from scipy.signal import butter, sosfilt
        
        nyquist = self.sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(4, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        
        return filtered
