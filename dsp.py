"""
DSP Frontend for Speech Enhancement
=====================================

A Python implementation of a DSP feature-extraction frontend similar to what
RNNoise and other hybrid DSP+neural speech enhancement systems use.

This extracts per-frame features from raw audio that can be fed into a small
neural network (GRU, LSTM, etc.) for mask/gain prediction.

Features extracted per frame:
  1. Band energies (in dB) on an ERB-like scale
  2. Pitch period (via autocorrelation)
  3. Pitch correlation (voicing strength)
  4. Pitch-filtered band energies (energy attributable to harmonics)
  5. First-order temporal derivatives of band energies

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile
from typing import NamedTuple


# =============================================================================
# 1. ERB (Equivalent Rectangular Bandwidth) Band Definition
# =============================================================================

def hz_to_erb(hz: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to ERB-rate scale (Glasberg & Moore 1990)."""
    return 9.265 * np.log(1 + hz / (24.7 * 9.265))


def erb_to_hz(erb: np.ndarray) -> np.ndarray:
    """Convert ERB-rate scale back to Hz."""
    return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


def compute_erb_bands(sr: int, n_fft: int, n_bands: int = 22):
    """
    Compute ERB-scale filterbank: a matrix that maps FFT bins to perceptual bands.

    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    n_fft : int
        FFT size.
    n_bands : int
        Number of ERB bands (typically 18-32; RNNoise uses 22).

    Returns
    -------
    filterbank : np.ndarray, shape (n_bands, n_fft // 2 + 1)
        Each row is a triangular filter for one ERB band.
    center_freqs : np.ndarray, shape (n_bands,)
        Center frequency of each band in Hz.
    """
    n_freqs = n_fft // 2 + 1
    freqs_hz = np.linspace(0, sr / 2, n_freqs)

    # Linearly space band edges on the ERB scale
    erb_low = hz_to_erb(np.array([20.0]))[0]       # ~20 Hz lower limit
    erb_high = hz_to_erb(np.array([sr / 2.0]))[0]
    erb_edges = np.linspace(erb_low, erb_high, n_bands + 2)
    hz_edges = erb_to_hz(erb_edges)

    center_freqs = hz_edges[1:-1]  # band centers

    filterbank = np.zeros((n_bands, n_freqs))

    for i in range(n_bands):
        lo, mid, hi = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]

        # Rising slope
        rising = (freqs_hz - lo) / max(mid - lo, 1e-8)
        # Falling slope
        falling = (hi - freqs_hz) / max(hi - mid, 1e-8)

        filterbank[i] = np.maximum(0, np.minimum(rising, falling))

    return filterbank, center_freqs


# =============================================================================
# 2. Pitch Detection via Autocorrelation
# =============================================================================

def estimate_pitch(frame: np.ndarray, sr: int,
                   pitch_min_hz: float = 60.0,
                   pitch_max_hz: float = 500.0):
    """
    Estimate pitch period and pitch correlation for a single frame
    using normalized autocorrelation.

    Parameters
    ----------
    frame : np.ndarray
        Audio samples for this frame (e.g., 320 samples for 20ms @ 16kHz).
    sr : int
        Sample rate.
    pitch_min_hz : float
        Minimum expected pitch (Hz). Sets the maximum lag to search.
    pitch_max_hz : float
        Maximum expected pitch (Hz). Sets the minimum lag to search.

    Returns
    -------
    pitch_period : int
        Estimated pitch period in samples. 0 if unvoiced.
    pitch_corr : float
        Normalized autocorrelation at the best pitch lag (0 to 1).
        High values (~0.7+) indicate voiced speech.
    """
    # Lag range in samples
    min_lag = max(int(sr / pitch_max_hz), 2)
    max_lag = min(int(sr / pitch_min_hz), len(frame) - 1)

    if max_lag <= min_lag:
        return 0, 0.0

    # Compute normalized autocorrelation for each candidate lag
    # R(lag) = sum(x[n] * x[n-lag]) / sqrt(sum(x[n]^2) * sum(x[n-lag]^2))
    frame = frame.astype(np.float64)
    n = len(frame)

    best_corr = -1.0
    best_lag = 0

    # Energy of the "current" part of the frame
    energy_current = np.sum(frame[max_lag:] ** 2)

    if energy_current < 1e-10:
        return 0, 0.0

    for lag in range(min_lag, max_lag + 1):
        # Cross-correlation at this lag
        x_current = frame[lag:]
        x_lagged = frame[:n - lag]

        # Make sure they're the same length
        min_len = min(len(x_current), len(x_lagged))
        x_current = x_current[:min_len]
        x_lagged = x_lagged[:min_len]

        cross_corr = np.sum(x_current * x_lagged)
        energy_lagged = np.sum(x_lagged ** 2)

        denom = np.sqrt(np.sum(x_current ** 2) * energy_lagged)
        if denom < 1e-10:
            continue

        corr = cross_corr / denom

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    # Threshold: if correlation is too low, consider it unvoiced
    pitch_period = best_lag if best_corr > 0.3 else 0
    pitch_corr = max(0.0, best_corr)

    return pitch_period, pitch_corr


def estimate_pitch_vectorized(frame: np.ndarray, sr: int,
                              pitch_min_hz: float = 60.0,
                              pitch_max_hz: float = 500.0):
    """
    Faster vectorized pitch estimation using numpy correlate.
    Same interface as estimate_pitch but much faster for large frames.
    """
    min_lag = max(int(sr / pitch_max_hz), 2)
    max_lag = min(int(sr / pitch_min_hz), len(frame) - 1)

    if max_lag <= min_lag:
        return 0, 0.0

    frame = frame.astype(np.float64)
    n = len(frame)

    # Full autocorrelation via numpy (faster than loop)
    # We only need positive lags from min_lag to max_lag
    autocorr = np.correlate(frame, frame, mode='full')
    # autocorr is length 2*n-1, centered at index n-1 (lag=0)
    autocorr = autocorr[n - 1:]  # positive lags only, shape (n,)

    # Compute running energy for normalization
    # energy[lag] = sum(frame[lag:]^2) -- but we approximate with frame energy
    energy_0 = autocorr[0]  # = sum(frame^2), this is energy at lag 0
    if energy_0 < 1e-10:
        return 0, 0.0

    # Normalized autocorrelation (approximate: using energy_0 for both terms)
    # More accurate would be per-lag energy, but this is standard in practice
    norm_autocorr = autocorr / energy_0

    # Search in the valid lag range
    search_range = norm_autocorr[min_lag:max_lag + 1]
    best_idx = np.argmax(search_range)
    best_lag = min_lag + best_idx
    best_corr = search_range[best_idx]

    pitch_period = best_lag if best_corr > 0.3 else 0
    pitch_corr = float(max(0.0, best_corr))

    return pitch_period, pitch_corr


# =============================================================================
# 3. Pitch Filtering (Comb Filter)
# =============================================================================

def pitch_comb_filter(frame_spectrum: np.ndarray, pitch_period: int,
                      n_fft: int, sr: int, comb_strength: float = 0.5):
    """
    Apply a comb filter in the frequency domain at the detected pitch.

    This reinforces harmonics and suppresses inter-harmonic content.
    The output represents "what the signal would look like if it were
    perfectly periodic at this pitch."

    Parameters
    ----------
    frame_spectrum : np.ndarray
        Complex STFT of the frame, shape (n_fft // 2 + 1,).
    pitch_period : int
        Detected pitch period in samples. If 0, returns zeros.
    n_fft : int
        FFT size.
    sr : int
        Sample rate.
    comb_strength : float
        Strength of the comb filter (0 = no filtering, 1 = full comb).

    Returns
    -------
    filtered_spectrum : np.ndarray
        The comb-filtered spectrum (complex), same shape as input.
    """
    if pitch_period == 0:
        return np.zeros_like(frame_spectrum)

    n_freqs = n_fft // 2 + 1
    freqs = np.arange(n_freqs) * sr / n_fft

    # Comb filter: H(f) = 1 + strength * cos(2*pi*f*T)
    # where T = pitch_period / sr is the pitch period in seconds
    # This has peaks at multiples of F0 = 1/T
    T = pitch_period / sr
    comb_response = 1.0 + comb_strength * np.cos(2 * np.pi * freqs * T)

    # Normalize so max gain = 1
    comb_response = comb_response / comb_response.max()

    filtered_spectrum = frame_spectrum * comb_response

    return filtered_spectrum


# =============================================================================
# 4. The Complete DSP Frontend
# =============================================================================

class FrameFeatures(NamedTuple):
    """Features extracted for a single audio frame."""
    band_energy_db: np.ndarray    # (n_bands,) - energy per ERB band in dB
    pitch_period: int             # pitch period in samples (0 = unvoiced)
    pitch_freq_hz: float          # pitch frequency in Hz (0 = unvoiced)
    pitch_corr: float             # pitch correlation / voicing strength (0-1)
    pitch_band_energy_db: np.ndarray  # (n_bands,) - pitch-filtered band energy
    delta_band_energy: np.ndarray     # (n_bands,) - temporal derivative of band energy


class DSPFrontend:
    """
    Complete DSP frontend for speech enhancement.

    Usage
    -----
    >>> frontend = DSPFrontend(sr=16000, frame_ms=20, hop_ms=10, n_bands=22)
    >>> features_list = frontend.process(audio_signal)
    >>> # Each element is a FrameFeatures namedtuple
    >>> feature_matrix = frontend.process_to_matrix(audio_signal)
    >>> # shape: (n_frames, n_features) — ready for a neural network
    """

    def __init__(self, sr: int = 16000, frame_ms: float = 20.0,
                 hop_ms: float = 10.0, n_bands: int = 22,
                 n_fft: int = None):
        """
        Parameters
        ----------
        sr : int
            Sample rate in Hz.
        frame_ms : float
            Frame length in milliseconds.
        hop_ms : float
            Hop (frame shift) in milliseconds.
        n_bands : int
            Number of ERB bands.
        n_fft : int or None
            FFT size. If None, uses the smallest power of 2 >= frame_size.
        """
        self.sr = sr
        self.frame_size = int(sr * frame_ms / 1000)
        self.hop_size = int(sr * hop_ms / 1000)
        self.n_bands = n_bands

        if n_fft is None:
            # Next power of 2 for efficient FFT
            self.n_fft = 1
            while self.n_fft < self.frame_size:
                self.n_fft *= 2
        else:
            self.n_fft = n_fft

        # Pre-compute the ERB filterbank
        self.filterbank, self.center_freqs = compute_erb_bands(
            sr, self.n_fft, n_bands
        )

        # Analysis window (Hann window for good spectral properties)
        self.window = np.hanning(self.frame_size)

        print(f"DSP Frontend initialized:")
        print(f"  Sample rate:  {sr} Hz")
        print(f"  Frame size:   {self.frame_size} samples ({frame_ms} ms)")
        print(f"  Hop size:     {self.hop_size} samples ({hop_ms} ms)")
        print(f"  FFT size:     {self.n_fft}")
        print(f"  ERB bands:    {n_bands}")
        print(f"  Band range:   {self.center_freqs[0]:.0f} - {self.center_freqs[-1]:.0f} Hz")

    def _compute_band_energy(self, power_spectrum: np.ndarray) -> np.ndarray:
        """
        Compute energy in each ERB band from the power spectrum.

        Returns energy in dB (with floor to avoid -inf).
        """
        band_energy = self.filterbank @ power_spectrum  # matrix multiply
        # Floor at a very small value to avoid log(0)
        band_energy = np.maximum(band_energy, 1e-10)
        return 10.0 * np.log10(band_energy)

    def _process_frame(self, frame: np.ndarray,
                       prev_band_energy_db: np.ndarray = None) -> FrameFeatures:
        """Extract all features from a single frame."""

        # --- Step 1: Window and FFT ---
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed, n=self.n_fft)
        power_spectrum = np.abs(spectrum) ** 2

        # --- Step 2: Band energies ---
        band_energy_db = self._compute_band_energy(power_spectrum)

        # --- Step 3: Pitch estimation ---
        pitch_period, pitch_corr = estimate_pitch_vectorized(
            frame, self.sr
        )
        pitch_freq = self.sr / pitch_period if pitch_period > 0 else 0.0

        # --- Step 4: Pitch-filtered band energies ---
        filtered_spectrum = pitch_comb_filter(
            spectrum, pitch_period, self.n_fft, self.sr
        )
        filtered_power = np.abs(filtered_spectrum) ** 2
        pitch_band_energy_db = self._compute_band_energy(filtered_power)

        # --- Step 5: Temporal derivative (delta features) ---
        if prev_band_energy_db is not None:
            delta = band_energy_db - prev_band_energy_db
        else:
            delta = np.zeros(self.n_bands)

        return FrameFeatures(
            band_energy_db=band_energy_db,
            pitch_period=pitch_period,
            pitch_freq_hz=pitch_freq,
            pitch_corr=pitch_corr,
            pitch_band_energy_db=pitch_band_energy_db,
            delta_band_energy=delta,
        )

    def process(self, audio: np.ndarray) -> list:
        """
        Process an entire audio signal and return a list of FrameFeatures.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio signal, float values in [-1, 1].

        Returns
        -------
        features : list of FrameFeatures
            One FrameFeatures per frame.
        """
        audio = audio.astype(np.float64)

        # Pad audio so we don't lose the tail
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        features = []
        prev_band_energy = None

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frame = audio[start:end]

            feat = self._process_frame(frame, prev_band_energy)
            features.append(feat)
            prev_band_energy = feat.band_energy_db

        return features

    def process_to_matrix(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio and return a 2D feature matrix ready for a neural network.

        Returns
        -------
        matrix : np.ndarray, shape (n_frames, n_features)
            Columns: [band_energy(22), pitch_period(1), pitch_corr(1),
                       pitch_band_energy(22), delta_band_energy(22)]
            Total: 22 + 1 + 1 + 22 + 22 = 68 features per frame.
        """
        features = self.process(audio)

        rows = []
        for f in features:
            row = np.concatenate([
                f.band_energy_db,                          # 22
                [f.pitch_period / self.sr * 1000],         # 1 (normalized to ms)
                [f.pitch_corr],                            # 1
                f.pitch_band_energy_db,                    # 22
                f.delta_band_energy,                       # 22
            ])
            rows.append(row)

        return np.array(rows)

    def get_feature_names(self) -> list:
        """Return human-readable names for each column in the feature matrix."""
        names = []
        for i in range(self.n_bands):
            names.append(f"band_energy_{i} ({self.center_freqs[i]:.0f}Hz)")
        names.append("pitch_period_ms")
        names.append("pitch_correlation")
        for i in range(self.n_bands):
            names.append(f"pitch_band_energy_{i} ({self.center_freqs[i]:.0f}Hz)")
        for i in range(self.n_bands):
            names.append(f"delta_band_energy_{i} ({self.center_freqs[i]:.0f}Hz)")
        return names


# =============================================================================
# 5. Demo / Usage Example
# =============================================================================

if __name__ == "__main__":
    # --- Generate a synthetic noisy speech-like signal for demo ---
    sr = 16000
    duration = 1.0  # seconds
    t = np.arange(int(sr * duration)) / sr

    # Simulate voiced speech: fundamental at 120 Hz + harmonics
    f0 = 120.0
    speech = np.zeros_like(t)
    for harmonic in range(1, 30):
        # Harmonics with decaying amplitude (speech-like spectral tilt)
        amplitude = 1.0 / harmonic
        speech += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
    speech = speech / np.max(np.abs(speech))  # normalize

    print(speech.shape)

    # Add noise at -10 dB SNR
    noise = np.random.randn(len(t)) * 0.3
    noisy = speech + noise

    # --- Run the DSP frontend ---
    frontend = DSPFrontend(sr=sr, frame_ms=20, hop_ms=10, n_bands=22)

    print(f"\nProcessing {duration}s of audio...")
    features = frontend.process(noisy)
    print(f"Extracted {len(features)} frames")

    # Look at one frame in the middle
    mid = len(features) // 2
    f = features[mid]
    print(f"\n--- Frame {mid} ---")
    print(f"  Pitch period:      {f.pitch_period} samples "
          f"({f.pitch_freq_hz:.1f} Hz)")
    print(f"  Pitch correlation: {f.pitch_corr:.3f}")
    print(f"  Band energies (dB): min={f.band_energy_db.min():.1f}, "
          f"max={f.band_energy_db.max():.1f}")

    # Get the full feature matrix
    matrix = frontend.process_to_matrix(noisy)
    print(f"\nFeature matrix shape: {matrix.shape}")
    print(f"  = ({matrix.shape[0]} frames, {matrix.shape[1]} features per frame)")
    print(f"\nFeature names (first 5):")
    for name in frontend.get_feature_names()[:5]:
        print(f"  {name}")
    print(f"  ... and {len(frontend.get_feature_names()) - 5} more")

    print("\n--- This matrix is ready to feed into your neural network! ---")
    print("Typical next step: train a small GRU/LSTM to predict per-band")
    print("gains (0-1) from these features, then apply gains to suppress noise.")