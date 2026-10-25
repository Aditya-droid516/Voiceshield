"""Unit tests for hand-made acoustic feature extraction."""
import numpy as np
import pytest

from hm.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    compute_cpps,
    compute_lfcc,
    extract_features,
    to_vector,
)


def test_feature_count_and_groups():
    assert len(FEATURE_NAMES) == 150
    assert len(FEATURE_GROUPS) == 9
    for grp_name, f_list in FEATURE_GROUPS.items():
        assert len(f_list) > 0


def test_extract_features_valid():
    sr = 16000
    t = np.arange(int(sr * 3.0)) / sr
    # Mix of sine waves + noise to simulate voice harmonics
    sig = (0.2 * np.sin(2 * np.pi * 220 * t) + 
           0.1 * np.sin(2 * np.pi * 440 * t) + 
           0.02 * np.random.randn(len(t))).astype(np.float32)

    feats = extract_features(sig, sr=sr)
    assert isinstance(feats, dict)
    assert len(feats) == len(FEATURE_NAMES)

    vec = to_vector(feats)
    assert vec.shape == (150,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_lfcc_and_cpps():
    sr = 16000
    sig = np.random.randn(32000).astype(np.float32) * 0.1
    lfcc = compute_lfcc(sig, sr=sr, n_coeffs=20)
    assert lfcc.shape[0] == 20
    assert not np.isnan(lfcc).any()

    cpps_m, cpps_s = compute_cpps(sig, sr=sr)
    assert isinstance(cpps_m, float)
    assert isinstance(cpps_s, float)
