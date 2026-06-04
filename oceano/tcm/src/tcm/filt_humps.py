"""
Claude.ai chat source (2026-03-17): https://claude.ai/public/artifacts/b3b8caa2-2c74-422d-99cc-b0c967b011e4

Hump removal for spectrograms with adaptive baseline and noise masking.

Pipeline per spectrum
---------------------
1. Estimate baseline over full spectrum (ALS by default).
2. Detect local maximum ("hump") in target frequency band [f_hump_min, f_hump_max].
3. Replace hump region with smooth baseline segment.
4. Optionally zero-out noise-dominated frequencies via SNR mask.

Baseline methods
----------------
- 'als'  : Asymmetric Least Squares — best for smoothly sloped spectra with narrow humps.
- 'poly' : Polynomial fit (peak-masked) — simple, fast.
- 'min'  : Morphological rolling minimum — handles step-like backgrounds.
- 'snip' : SNIP (log-space iterative) — classic for gamma spectra.

References
----------
Eilers & Boelens (2005) — Baseline Correction with Asymmetric Least Squares.
Ryan et al. (1988) — SNIP algorithm.
"""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

log = logging.getLogger(__name__)

# ── Baseline estimators ───────────────────────────────────────────────────────

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares smoothing baseline.

    Parameters
    ----------
    lam   : smoothness penalty (larger → smoother).
    p     : asymmetry weight (0.001–0.1); small values penalise positive residuals.
    niter : number of reweighting iterations.
    """
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).toarray()
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    DTD = D.T @ D
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = diags(w)
        z = spsolve(W + lam * DTD, w * y)
        w = np.where(y > z, p, 1 - p)
    log.debug("ALS baseline computed, lam=%.0e p=%.3f iters=%d", lam, p, niter)
    return z


def baseline_poly(y: np.ndarray, deg: int = 3, peak_threshold: float = 0.1) -> np.ndarray:
    """
    Polynomial baseline fitted on peak-masked spectrum.

    Parameters
    ----------
    deg             : polynomial degree.
    peak_threshold  : fraction of max amplitude above which points are excluded.
    """
    x = np.arange(len(y))
    threshold = np.max(y) * peak_threshold
    peaks, _ = find_peaks(y, height=threshold * 0.5)
    mask = (y < threshold)
    mask[peaks] = False
    coeffs = np.polyfit(x[mask], y[mask], deg) if mask.sum() > deg else np.polyfit(x, y, deg)
    log.debug("Poly baseline computed, deg=%d masked_pts=%d", deg, mask.sum())
    return np.polyval(coeffs, x)


def baseline_morphological(y: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling-minimum + Gaussian smoothing baseline."""
    bl = gaussian_filter1d(minimum_filter1d(y, size=window, mode='reflect'), sigma=window // 4)
    log.debug("Morphological baseline computed, window=%d", window)
    return bl


def baseline_snip(y: np.ndarray, niter: int = 10, m: int = 5) -> np.ndarray:
    """
    SNIP algorithm (vectorised) in linear space.

    Parameters
    ----------
    niter : number of iterations (controls how wide a hump can be absorbed).
    m     : half-window for neighbour comparison.
    """
    z = np.maximum(y, np.finfo(float).eps).copy()
    for _ in range(niter):
        left  = np.concatenate([z[:m],  z[:-m]])
        right = np.concatenate([z[m:],  z[-m:]])
        z = np.minimum(z, np.sqrt(left * right))
    log.debug("SNIP baseline computed, niter=%d m=%d", niter, m)
    return z


# ── Noise estimation (from spectral shape) ────────────────────────────────────

def estimate_noise_from_spectrum(
    f: np.ndarray,
    S: np.ndarray,
    df_smooth: int = 5,
    slope_window: int = 7,
    flat_slope_thresh: float = 0.2,
) -> tuple[float, np.ndarray]:
    """
    Estimate sensor noise level from spectral shape (adaptive, no fixed tail).

    Detects the noise floor as the region where d(log S)/d(log f) ≈ 0.
    Falls back to min of smoothed spectrum when no flat region is found.

    Returns
    -------
    noise_level : float
    mask_noise  : bool ndarray — True where spectrum is noise-dominated.
    """
    eps = np.finfo(float).eps
    logf, logS = np.log(f), np.log(np.maximum(S, eps))
    kernel = np.ones(df_smooth) / df_smooth
    logS_s = np.convolve(logS, kernel, mode='same')
    slope = np.convolve(np.gradient(logS_s, logf), np.ones(slope_window) / slope_window, mode='same')
    mask_noise = np.abs(slope) < flat_slope_thresh
    noise_level = float(np.median(S[mask_noise])) if mask_noise.any() else float(np.exp(logS_s).min())
    log.debug("Noise estimate: level=%.3e flat_pts=%d", noise_level, mask_noise.sum())
    return noise_level, mask_noise


def fit_noise_power_law(f: np.ndarray, S: np.ndarray, mask_noise: np.ndarray) -> tuple[float, float]:
    """Fit S_n = a * f^b to noise-dominated region. Returns (a, b)."""
    b, loga = np.polyfit(np.log(f[mask_noise]), np.log(S[mask_noise]), 1)
    return float(np.exp(loga)), float(b)


def snr_mask(
    f: np.ndarray,
    S: np.ndarray,
    noise_spectrum: np.ndarray,
    snr_threshold: float = 5.0,
    fmin: float = 0.04,
) -> np.ndarray:
    """Bool mask: True where SNR >= snr_threshold and f >= fmin."""
    return (f >= fmin) & (S / noise_spectrum >= snr_threshold)


# ── Core hump removal ─────────────────────────────────────────────────────────

def remove_hump(
    spectrogram: np.ndarray,
    f: np.ndarray,
    f_hump_min: float,
    f_hump_max: float,
    baseline_fn: callable,
    baseline_kw: dict | None = None,
    prominence_factor: float = 0.1,
    apply_snr_mask: bool = False,
    snr_threshold: float = 5.0,
    fmin: float = 0.04,
    return_baselines: bool = False,
) -> tuple[np.ndarray, list[dict], np.ndarray | None]:
    """
    Remove a local maximum ("hump") from each spectrum in a spectrogram.

    Parameters
    ----------
    spectrogram     : (n_spectra, n_freqs) array.
    f               : frequency axis, length n_freqs.
    f_hump_min      : lower bound of hump search region [Hz].
    f_hump_max      : upper bound of hump search region [Hz].
    baseline_fn     : the baseline estimator function, default to `baseline_als`. See also baseline_poly,
    baseline_min, baseline_snip that can be used
    baseline_kw     : extra kwargs forwarded to `baseline_fn`.
    prominence_factor: peak detection threshold relative to corrected-spectrum max.
    apply_snr_mask  : if True, zero-out noise-dominated frequencies after hump removal.
    snr_threshold   : SNR cutoff for noise masking.
    fmin            : minimum frequency for SNR mask.
    return_baselines: if True, return baselines array as third element.

    Returns
    -------
    cleaned    : (n_spectra, n_freqs) corrected spectrogram.
    hump_info  : list of per-spectrum dicts with hump location / amplitude / bounds.
    baselines  : (n_spectra, n_freqs) array if return_baselines else None.
    """
    baseline_kw = baseline_kw or {}
    f = np.asarray(f)
    i_min, i_max = np.searchsorted(f, [f_hump_min, f_hump_max])
    i_max = min(i_max, spectrogram.shape[1] - 1)
    log.info("Hump search band: %.3f–%.3f Hz → indices %d–%d", f_hump_min, f_hump_max, i_min, i_max)

    cleaned = spectrogram.copy()
    hump_info, baselines = [], ([] if return_baselines else None)

    for idx, spectrum in enumerate(cleaned):
        bl = baseline_fn(spectrum, **baseline_kw)
        if return_baselines:
            baselines.append(bl.copy())

        corrected = spectrum - bl
        peak_threshold = np.max(corrected) * prominence_factor
        raw_peaks, _ = find_peaks(corrected[i_min:i_max], prominence=peak_threshold)
        peaks = raw_peaks + i_min

        info: dict = {'spectrum_idx': idx, 'hump_position': None}

        if len(peaks):
            hump_idx = peaks[np.argmax(corrected[peaks])]
            log.debug("[%d] hump candidate at f=%.4f Hz (idx=%d)", idx, f[hump_idx], hump_idx)

            # Boundary: walk left/right until corrected ≤ 0 or hits band edge
            left_idx = (
                i_min + (np.where(corrected[i_min:hump_idx] <= 0)[0] or [0])[-1]
            )
            right_candidates = np.where(corrected[hump_idx:i_max] <= 0)[0]
            right_idx = hump_idx + (right_candidates[0] if len(right_candidates) else (i_max - hump_idx) // 2)

            # Build smooth replacement: baseline segment with boundary-matched endpoints
            seg = bl[left_idx:right_idx + 1].copy()
            n = len(seg)
            if n > 1:
                delta = (bl[right_idx] - bl[left_idx]) / n - (seg[-1] - seg[0]) / n
                seg += np.linspace(0, delta * n, n)
            if n > 5:
                seg = savgol_filter(seg, min(5, n // 2 * 2 + 1), 2)

            orig = spectrogram[idx]
            cleaned[idx, left_idx:right_idx + 1] = seg
            info.update({
                'hump_position':    int(hump_idx),
                'hump_freq':        float(f[hump_idx]),
                'hump_amplitude':   float(orig[hump_idx]),
                'baseline_at_hump': float(bl[hump_idx]),
                'left_bound':       int(left_idx),
                'right_bound':      int(right_idx),
                'removed_area':     float(np.sum(orig[left_idx:right_idx + 1] - seg)),
            })
            log.info(
                "[%d] hump removed: f=%.4f Hz, bounds=[%d, %d], area=%.3e",
                *(idx, f[hump_idx], left_idx, right_idx, info["removed_area"]),
            )

        if apply_snr_mask:
            noise_level, mask_noise = estimate_noise_from_spectrum(f, cleaned[idx])
            a, b = fit_noise_power_law(f, cleaned[idx], mask_noise)
            valid = snr_mask(f, cleaned[idx], a * f ** b, snr_threshold=snr_threshold, fmin=fmin)
            cleaned[idx, ~valid] = 0.0
            info['snr_mask_zeroed'] = int((~valid).sum())
            log.debug("[%d] SNR mask zeroed %d freqs", idx, info['snr_mask_zeroed'])

        hump_info.append(info)

    log.info(
        "Done. %d/%d spectra had hump detected.",
        sum(1 for h in hump_info if h["hump_position"] is not None),
        len(hump_info),
    )
    return cleaned, hump_info, (np.array(baselines) if return_baselines else None)
