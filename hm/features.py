"""Hand-crafted acoustic feature extraction for VoiceShield.

Extracts explainable speech characteristics:
- Spectral texture (MFCCs, deltas, spectral moments, contrast, flux)
- Vocoder traces (LFCCs, high-freq flatness, autocorrelation peaks)
- High-frequency detail (sub-band energies, spectral tilt, cut-off frequency)
- Pitch & prosody (F0 statistics, semitones, jumps) via Praat
- Jitter & shimmer (local, RAP, PPQ5, APQ3, APQ5) via Praat
- Harmonics vs noise (HNR, CPPS)
- Formants F1-F3 (frequencies, bandwidths, frame-to-frame steps) via Praat
- Rhythm & pauses (syllable rate 4 Hz modulation, pause metrics, onset rate)
- Room & noise floor (SNR, noise floor, quiet-frame spectrum)

All features are computed on the cleaned audio BEFORE tiling to 4s,
preserving authentic temporal and prosodic dynamics.
"""

import math
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from scipy import fftpack, signal
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocess import preprocess_audio, AudioError

# Define Linear Filterbank for LFCC
def _linear_filterbank(n_filters=20, n_fft=512, sr=16000, low_freq=0, high_freq=8000):
    """Linearly-spaced triangular filterbank spanning low_freq to high_freq."""
    freq_bins = np.linspace(low_freq, high_freq, n_filters + 2)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fbank = np.zeros((n_filters, len(fft_freqs)), dtype=np.float32)

    for i in range(n_filters):
        f_left = freq_bins[i]
        f_center = freq_bins[i + 1]
        f_right = freq_bins[i + 2]

        left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-9)
        right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-9)
        triangle = np.maximum(0.0, np.minimum(left_slope, right_slope))
        fbank[i] = triangle

    return fbank


LFCC_FBANK = _linear_filterbank(n_filters=20, n_fft=512, sr=16000)


def compute_lfcc(y, sr=16000, n_filters=20, n_coeffs=20, n_fft=512, hop_length=256):
    """Linear Frequency Cepstral Coefficients (LFCC) - crucial for vocoder artifact detection."""
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    fbank_energies = np.dot(LFCC_FBANK, D)
    log_energies = np.log(np.maximum(fbank_energies, 1e-10))
    lfcc = fftpack.dct(log_energies, axis=0, type=2, norm="ortho")[:n_coeffs]
    return lfcc


def compute_cpps(y, sr=16000, frame_len=1024, hop_len=512):
    """Smoothed Cepstral Peak Prominence (CPPS)."""
    # Framing
    num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    prominences = []
    min_quef = int(sr / 600)  # 600 Hz ceiling
    max_quef = int(sr / 75)   # 75 Hz floor

    for i in range(num_frames):
        frame = y[i * hop_len : i * hop_len + frame_len] * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame, n=frame_len))
        log_spec = np.log(spec + 1e-9)
        cepstrum = np.real(np.fft.irfft(log_spec))
        
        if max_quef < len(cepstrum):
            sub_cep = cepstrum[min_quef:max_quef]
            peak_val = np.max(sub_cep)
            # Baseline regression on the quefrency range
            x_vals = np.arange(min_quef, max_quef)
            fit = np.polyfit(x_vals, sub_cep, 1)
            baseline = np.polyval(fit, min_quef + np.argmax(sub_cep))
            prominences.append(float(peak_val - baseline))

    if not prominences:
        return 0.0, 0.0
    return float(np.mean(prominences)), float(np.std(prominences))


def extract_features(y, sr=16000):
    """Extract full vector of ~140 explainable acoustic features from audio array."""
    y = np.asarray(y, dtype=np.float32)
    dur = float(len(y) / sr)
    if dur < 0.1:
        raise AudioError("Clip too short for feature extraction.")

    feats = {}

    # ---------------------------------------------------------
    # 1. SPECTRAL TEXTURE
    # ---------------------------------------------------------
    # MFCC (20 mean, 20 std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    for i in range(20):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # Delta MFCC (20 mean abs)
    d_mfcc = librosa.feature.delta(mfcc)
    for i in range(20):
        feats[f"delta_mfcc_{i+1}_mean_abs"] = float(np.mean(np.abs(d_mfcc[i])))

    # Spectral moments & shapes
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=256)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=256)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=256)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=256)[0]

    feats["spec_centroid_mean"] = float(np.mean(cent))
    feats["spec_centroid_std"] = float(np.std(cent))
    feats["spec_bandwidth_mean"] = float(np.mean(bw))
    feats["spec_bandwidth_std"] = float(np.std(bw))
    feats["spec_rolloff_mean"] = float(np.mean(rolloff))
    feats["spec_rolloff_std"] = float(np.std(rolloff))
    feats["spec_flatness_mean"] = float(np.mean(flat))
    feats["spec_flatness_std"] = float(np.std(flat))
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    # Spectral Contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=256)
    for b in range(7):
        feats[f"spec_contrast_band_{b+1}_mean"] = float(np.mean(contrast[b]))

    # Spectral Flux
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
    flux = np.sqrt(np.mean((S[:, 1:] - S[:, :-1]) ** 2, axis=0)) if S.shape[1] > 1 else np.array([0.0])
    feats["spec_flux_mean"] = float(np.mean(flux))
    feats["spec_flux_std"] = float(np.std(flux))

    # ---------------------------------------------------------
    # 2. VOCODER TRACES
    # ---------------------------------------------------------
    # 20 LFCC means
    lfcc = compute_lfcc(y, sr=sr, n_filters=20, n_coeffs=20)
    for i in range(20):
        feats[f"lfcc_{i+1}_mean"] = float(np.mean(lfcc[i]))

    # Flatness in 4-8 kHz band
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    high_mask = (freqs >= 4000) & (freqs <= 8000)
    if high_mask.sum() > 0:
        high_spec = S[high_mask, :]
        # Flatness = geometric mean / arithmetic mean
        log_spec = np.log(high_spec + 1e-12)
        geom = np.exp(np.mean(log_spec, axis=0))
        arith = np.mean(high_spec, axis=0) + 1e-12
        hf_flat = geom / arith
        feats["hf_flatness_mean"] = float(np.mean(hf_flat))
        feats["hf_flatness_std"] = float(np.std(hf_flat))
    else:
        feats["hf_flatness_mean"] = 0.0
        feats["hf_flatness_std"] = 0.0

    # Autocorrelation peak strength & variation
    # Frame size 25ms, hop 10ms
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    min_lag = int(sr / 500)  # 500 Hz
    max_lag = int(sr / 65)   # 65 Hz
    ac_peaks = []
    for s_idx in range(0, len(y) - win, hop):
        seg = y[s_idx : s_idx + win]
        r = np.correlate(seg, seg, mode="full")[win - 1 :]
        if r[0] > 1e-8:
            r_norm = r / r[0]
            if max_lag < len(r_norm):
                ac_peaks.append(float(np.max(r_norm[min_lag:max_lag])))
    if ac_peaks:
        feats["autocorr_peak_mean"] = float(np.mean(ac_peaks))
        feats["autocorr_peak_std"] = float(np.std(ac_peaks))
    else:
        feats["autocorr_peak_mean"] = 0.0
        feats["autocorr_peak_std"] = 0.0

    # ---------------------------------------------------------
    # 3. HIGH-FREQUENCY DETAIL
    # ---------------------------------------------------------
    # Log energy in 6 bands: 0-0.5, 0.5-1, 1-2, 2-4, 4-6, 6-8 kHz
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]
    power = S ** 2
    band_powers = []
    for idx, (f_lo, f_hi) in enumerate(bands):
        m = (freqs >= f_lo) & (freqs < f_hi)
        bp = float(np.sum(power[m, :])) if m.sum() > 0 else 0.0
        feats[f"log_energy_band_{idx+1}"] = float(np.log10(bp + 1e-9))
        band_powers.append(bp)

    # High-to-low energy ratio (>2kHz vs <2kHz)
    low_e = band_powers[0] + band_powers[1] + band_powers[2] + 1e-9
    high_e = band_powers[3] + band_powers[4] + band_powers[5] + 1e-9
    feats["high_to_low_energy_ratio"] = float(high_e / low_e)

    # Spectral tilt (slope of log power across frequency bins)
    mean_power_spectrum = np.mean(power, axis=1) + 1e-12
    log_p = np.log10(mean_power_spectrum)
    tilt_slope = float(np.polyfit(freqs / 1000.0, log_p, 1)[0])
    feats["spectral_tilt"] = tilt_slope

    # Estimated cut-off frequency (-30 dB from max power bin)
    max_p = np.max(mean_power_spectrum)
    cutoff_thresh = max_p * 1e-3
    above_thresh = np.where(mean_power_spectrum > cutoff_thresh)[0]
    feats["estimated_cutoff_freq"] = float(freqs[above_thresh[-1]]) if len(above_thresh) > 0 else 8000.0

    # ---------------------------------------------------------
    # 4. PITCH, JITTER, SHIMMER, HNR, FORMANTS (Praat-Parselmouth)
    # ---------------------------------------------------------
    # Safe defaults in case Praat encounters an unvoiced/degenerate frame
    feats["pitch_voiced_ratio"] = 0.0
    feats["pitch_mean_semitones"] = 0.0
    feats["pitch_std_semitones"] = 0.0
    feats["pitch_range_semitones"] = 0.0
    feats["pitch_mean_jump"] = 0.0
    feats["pitch_large_jumps_rate"] = 0.0

    feats["jitter_local"] = 0.0
    feats["jitter_rap"] = 0.0
    feats["jitter_ppq5"] = 0.0
    feats["shimmer_local"] = 0.0
    feats["shimmer_apq3"] = 0.0
    feats["shimmer_apq5"] = 0.0

    feats["hnr_mean"] = 0.0
    feats["hnr_std"] = 0.0

    feats["f1_mean"] = 500.0
    feats["f1_std"] = 50.0
    feats["f1_step_mean"] = 20.0
    feats["f2_mean"] = 1500.0
    feats["f2_std"] = 100.0
    feats["f2_step_mean"] = 30.0
    feats["f3_mean"] = 2500.0
    feats["f3_std"] = 150.0
    feats["f3_step_mean"] = 40.0

    try:
        sound = parselmouth.Sound(values=y, sampling_frequency=sr)
        
        # Pitch F0
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        pitch_vals = pitch.selected_array["frequency"]
        voiced = pitch_vals[pitch_vals > 0]
        feats["pitch_voiced_ratio"] = float(len(voiced) / max(len(pitch_vals), 1))
        
        if len(voiced) > 2:
            st = 12.0 * np.log2(voiced / 100.0)  # semitones relative to 100 Hz
            feats["pitch_mean_semitones"] = float(np.mean(st))
            feats["pitch_std_semitones"] = float(np.std(st))
            feats["pitch_range_semitones"] = float(np.percentile(st, 95) - np.percentile(st, 5))
            
            # Frame jumps across consecutive voiced frames
            jumps = np.abs(np.diff(st))
            feats["pitch_mean_jump"] = float(np.mean(jumps)) if len(jumps) > 0 else 0.0
            feats["pitch_large_jumps_rate"] = float(np.mean(jumps > 2.0)) if len(jumps) > 0 else 0.0

        # PointProcess for Jitter and Shimmer
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        n_points = parselmouth.praat.call(pp, "Get number of points")
        if n_points > 5:
            feats["jitter_local"] = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            feats["jitter_rap"] = float(parselmouth.praat.call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
            feats["jitter_ppq5"] = float(parselmouth.praat.call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
            feats["shimmer_local"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            feats["shimmer_apq3"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            feats["shimmer_apq5"] = float(parselmouth.praat.call([sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        # Harmonics-to-Noise Ratio (HNR)
        harm = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hnr_vals = harm.values[harm.values > -100]
        if len(hnr_vals) > 0:
            feats["hnr_mean"] = float(np.mean(hnr_vals))
            feats["hnr_std"] = float(np.std(hnr_vals))

        # Formants F1-F3
        formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
        t_grid = np.arange(0.02, dur - 0.02, 0.01)
        f1s, f2s, f3s = [], [], []
        for t_val in t_grid:
            v1 = formants.get_value_at_time(1, t_val)
            v2 = formants.get_value_at_time(2, t_val)
            v3 = formants.get_value_at_time(3, t_val)
            if not np.isnan(v1): f1s.append(v1)
            if not np.isnan(v2): f2s.append(v2)
            if not np.isnan(v3): f3s.append(v3)
            
        if len(f1s) > 2:
            feats["f1_mean"] = float(np.mean(f1s))
            feats["f1_std"] = float(np.std(f1s))
            feats["f1_step_mean"] = float(np.mean(np.abs(np.diff(f1s))))
        if len(f2s) > 2:
            feats["f2_mean"] = float(np.mean(f2s))
            feats["f2_std"] = float(np.std(f2s))
            feats["f2_step_mean"] = float(np.mean(np.abs(np.diff(f2s))))
        if len(f3s) > 2:
            feats["f3_mean"] = float(np.mean(f3s))
            feats["f3_std"] = float(np.std(f3s))
            feats["f3_step_mean"] = float(np.mean(np.abs(np.diff(f3s))))

    except Exception:
        pass  # keep default values

    # Smoothed Cepstral Peak Prominence (CPPS)
    cpps_mean, cpps_std = compute_cpps(y, sr=sr)
    feats["cpps_mean"] = cpps_mean
    feats["cpps_std"] = cpps_std

    # ---------------------------------------------------------
    # 5. RHYTHM AND PAUSES
    # ---------------------------------------------------------
    frame_len_rhythm = 1024
    hop_len_rhythm = 256
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len_rhythm, hop_length=hop_len_rhythm)[0]
    
    # Loudness variation
    mean_rms = np.mean(rms_frames) + 1e-9
    feats["loudness_variation"] = float(np.std(rms_frames) / mean_rms)

    # Pauses (< -25 dB relative to max frame RMS)
    max_rms = np.max(rms_frames) + 1e-9
    db_frames = 20 * np.log10(np.maximum(rms_frames / max_rms, 1e-5))
    is_pause = (db_frames < -25.0).astype(int)
    feats["pause_ratio"] = float(np.mean(is_pause))

    # Transitions to pause
    pause_transitions = np.diff(is_pause) == 1
    feats["pauses_per_second"] = float(np.sum(pause_transitions) / dur)

    # Mean pause length
    pause_runs = []
    cur_run = 0
    dt_frame = hop_len_rhythm / sr
    for p in is_pause:
        if p == 1:
            cur_run += 1
        else:
            if cur_run > 0:
                pause_runs.append(cur_run * dt_frame)
                cur_run = 0
    if cur_run > 0:
        pause_runs.append(cur_run * dt_frame)
    feats["mean_pause_length"] = float(np.mean(pause_runs)) if pause_runs else 0.0

    # Onset rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_len_rhythm)
    feats["onset_rate"] = float(len(onsets) / dur)

    # Voiced segment length
    # Voiced segment lengths from is_pause (active speech chunks)
    speech_runs = []
    c_speech = 0
    for p in is_pause:
        if p == 0:
            c_speech += 1
        else:
            if c_speech > 0:
                speech_runs.append(c_speech * dt_frame)
                c_speech = 0
    if c_speech > 0:
        speech_runs.append(c_speech * dt_frame)
    feats["voiced_segment_mean_length"] = float(np.mean(speech_runs)) if speech_runs else 0.0

    # 4 Hz syllable-rate modulation energy
    env = np.abs(signal.hilbert(y))
    # Resample envelope to 100 Hz
    env_ds = signal.resample(env, int(dur * 100))
    env_ds -= np.mean(env_ds)
    env_spec = np.abs(np.fft.rfft(env_ds)) ** 2
    env_freqs = np.fft.rfftfreq(len(env_ds), d=1.0 / 100.0)
    syl_mask = (env_freqs >= 3.0) & (env_freqs <= 5.5)  # 3-5.5 Hz syllable band
    tot_env_pow = np.sum(env_spec) + 1e-9
    feats["syllable_4hz_modulation_energy"] = float(np.sum(env_spec[syl_mask]) / tot_env_pow)

    # ---------------------------------------------------------
    # 6. ROOM AND NOISE FLOOR
    # ---------------------------------------------------------
    # Noise floor: 10th percentile of RMS
    noise_floor = float(np.percentile(rms_frames, 10))
    speech_peak = float(np.percentile(rms_frames, 90))
    feats["noise_floor_estimate"] = noise_floor
    feats["snr_estimate_db"] = float(20 * np.log10(max(speech_peak, 1e-6) / max(noise_floor, 1e-6)))

    # Spectrum of quietest frames (bottom 10%)
    quiet_idx = np.where(rms_frames <= np.percentile(rms_frames, 10))[0]
    if len(quiet_idx) > 0 and S.shape[1] > 0:
        quiet_idx = quiet_idx[quiet_idx < S.shape[1]]
        if len(quiet_idx) > 0:
            quiet_power = np.mean(S[:, quiet_idx] ** 2, axis=1)
        else:
            quiet_power = np.mean(S ** 2, axis=1)
    else:
        quiet_power = np.mean(S ** 2, axis=1)

    # 4 bands of quiet spectrum: 0-1k, 1-2k, 2-4k, 4-8k
    q_bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    for idx, (lo, hi) in enumerate(q_bands):
        m = (freqs >= lo) & (freqs < hi)
        qp = float(np.sum(quiet_power[m])) if m.sum() > 0 else 0.0
        feats[f"quiet_spectrum_band_{idx+1}"] = float(np.log10(qp + 1e-9))

    # Sanitize any NaNs or infinities
    for k, v in feats.items():
        if math.isnan(v) or math.isinf(v):
            feats[k] = 0.0

    return feats


def extract_from_file(path, strict=False):
    """Load cleaned audio without tiling to 4s, and extract feature dictionary."""
    y, sr = preprocess_audio(path, strict=strict)
    return extract_features(y, sr)


# Canonical ordered list of feature names
_dummy_signal = np.sin(np.linspace(0, 100 * np.pi, 32000)).astype(np.float32)
_dummy_feats = extract_features(_dummy_signal, sr=16000)
FEATURE_NAMES = list(_dummy_feats.keys())

FEATURE_GROUPS = {
    "spectral_texture": [k for k in FEATURE_NAMES if "mfcc" in k or "spec_" in k or "zcr" in k],
    "vocoder_traces": [k for k in FEATURE_NAMES if "lfcc" in k or "hf_flat" in k or "autocorr" in k],
    "high_frequency": [k for k in FEATURE_NAMES if "log_energy" in k or "high_to_low" in k or "tilt" in k or "cutoff" in k],
    "pitch_prosody": [k for k in FEATURE_NAMES if "pitch" in k],
    "jitter_shimmer": [k for k in FEATURE_NAMES if "jitter" in k or "shimmer" in k],
    "harmonics_noise": [k for k in FEATURE_NAMES if "hnr" in k or "cpps" in k],
    "formants": [k for k in FEATURE_NAMES if k.startswith("f1_") or k.startswith("f2_") or k.startswith("f3_")],
    "rhythm_pauses": [k for k in FEATURE_NAMES if "loudness" in k or "pause" in k or "onset" in k or "syllable" in k or "voiced_segment" in k],
    "room_noise": [k for k in FEATURE_NAMES if "noise_floor" in k or "snr" in k or "quiet_spectrum" in k],
}


def to_vector(feat_dict):
    """Convert dictionary to ordered 1D numpy vector according to FEATURE_NAMES."""
    return np.array([feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)


def extract_dataset(pairs, out_path, limit=None, save_every=250):
    """Resumable feature extraction across audio clips saved to .npz file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs)[:limit] if limit else list(pairs)

    X, y, paths = [], [], []
    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        X, y, paths = list(d["X"]), list(d["y"]), list(d["paths"])
    done = set(paths)
    todo = [(p, l) for p, l in pairs if str(p) not in done]
    print(f"[{out_path.name}] {len(done)} already extracted, {len(todo)} to process.")
    if not todo:
        return out_path

    def save():
        np.savez_compressed(
            out_path,
            X=np.stack(X) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            y=np.array(y, dtype=np.int32),
            paths=np.array(paths),
            feature_names=np.array(FEATURE_NAMES),
        )

    skipped = 0
    t0 = time.time()
    for n, (p, label) in enumerate(tqdm(todo, desc=f"Extracting {out_path.stem}"), 1):
        try:
            fd = extract_from_file(p, strict=False)
            vec = to_vector(fd)
            X.append(vec)
            y.append(int(label))
            paths.append(str(p))
        except (AudioError, Exception) as e:
            skipped += 1
            continue

        if n % save_every == 0:
            save()

    if X:
        save()
    elapsed = time.time() - t0
    print(f"Saved {len(X)} clips to {out_path} ({skipped} skipped, {elapsed:.1f}s total, {elapsed/max(len(todo),1)*1000:.1f}ms/clip)")
    return out_path


if __name__ == "__main__":
    print(f"Total Hand-Made Features: {len(FEATURE_NAMES)}")
    for grp, f_list in FEATURE_GROUPS.items():
        print(f"  {grp:<18}: {len(f_list)} features")
    # Quick sanity check on 1 file
    test_file = ROOT / "data" / "raw" / "release_in_the_wild" / "44.wav"
    t_start = time.time()
    feats = extract_from_file(test_file)
    dur_ms = (time.time() - t_start) * 1000
    vec = to_vector(feats)
    print(f"\nExtracted from {test_file.name} in {dur_ms:.1f}ms: shape {vec.shape}, NaNs: {np.isnan(vec).sum()}")
    print("Sample features:")
    for k in list(feats.keys())[:10]:
        print(f"  {k:<30}: {feats[k]:.4f}")
