# ===== Wibbly Wobbly Bois =====
import math

import numpy as np

from sneksynth.envelope import adsr_env_fitted


def osc_sine(freq: float, length_s: float, sr: int) -> np.ndarray:
    """
    Beep.
    """
    n = int(length_s * sr)
    t = np.linspace(0.0, length_s, n, endpoint=False)
    return np.sin(2.0 * math.pi * freq * t)


def osc_square(freq: float, length_s: float, sr: int) -> np.ndarray:
    """
    Boop.
    """
    s = osc_sine(freq, length_s, sr)
    return np.sign(s)


def osc_triangle(
    freq: float, length_s: float, sr: int, partials: int = 15
) -> np.ndarray:
    """
    Band-limited-ish triangle via additive synthesis of odd harmonics (1/k^2 amplitude).
    """
    n = int(length_s * sr)
    t = np.linspace(0.0, length_s, n, endpoint=False)
    y = np.zeros_like(t)
    for k in range(1, 2 * partials, 2):
        y += (
            ((-1) ** ((k - 1) // 2))
            * (1.0 / (k * k))
            * np.sin(2.0 * math.pi * k * freq * t)
        )
    y = y * (8.0 / (math.pi**2))
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def detuned_saws(
    freq: float, length_s: float, sr: int, detune_cents: float = 7.0, partials: int = 12
) -> np.ndarray:
    """
    Detuned twin band-limited saws, summed and normalized.
    """
    n = int(length_s * sr)
    t = np.linspace(0.0, length_s, n, endpoint=False)
    cents = detune_cents / 1200.0
    f1 = freq * (2.0**cents)
    f2 = freq * (2.0**-cents)

    def bandlimited_saw(f: float) -> np.ndarray:
        wave = np.zeros_like(t)
        for k in range(1, partials + 1):
            wave += (1.0 / k) * np.sin(2.0 * math.pi * k * f * t)
        peak = np.max(np.abs(wave))
        return wave / peak if peak > 0.0 else wave

    w1 = bandlimited_saw(f1)
    w2 = bandlimited_saw(f2)
    # Why can't you be normal?
    mix = 0.5 * (w1 + w2)
    peak = np.max(np.abs(mix))
    return mix / peak if peak > 0.0 else mix


# ===== Drums =====
def drum_kick(
    length_s: float, sr: int, f_start: float = 100.0, f_end: float = 40.0
) -> np.ndarray:
    """
    Kick = pitch-swept sine + smol click.
    """
    n = int(length_s * sr)
    t = np.linspace(0.0, length_s, n, endpoint=False) # noqa: F841 # type: ignore
    # Exponential sweep
    k = (f_end / f_start) ** (1.0 / max(1, n))
    freqs = f_start * (k ** np.arange(n))
    phase = 2.0 * math.pi * np.cumsum(freqs) / sr
    body = np.sin(phase)
    body_env = adsr_env_fitted(n, sr, a=0.001, d=0.06, s=0.3, r=0.08)
    # Click
    click = np.zeros_like(body)
    click_len = max(4, int(0.002 * sr))
    click[:click_len] = 1.0
    click_env = adsr_env_fitted(n, sr, a=0.0005, d=0.01, s=0.0, r=0.02)
    sig = 0.95 * body * body_env + 0.15 * click * click_env
    # Safety norm
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig


def drum_snare(length_s: float, sr: int) -> np.ndarray:
    """
    Snare = short hiss + low sine thunk.
    """
    n = int(length_s * sr)
    noise = np.random.uniform(-1.0, 1.0, n)
    noise_env = adsr_env_fitted(n, sr, a=0.0005, d=0.12, s=0.0, r=0.08)
    thunk = osc_sine(180.0, length_s, sr) * adsr_env_fitted(
        n, sr, a=0.001, d=0.05, s=0.0, r=0.05
    )
    sig = 0.8 * noise * noise_env + 0.25 * thunk
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig


def drum_hat(length_s: float, sr: int) -> np.ndarray:
    """
    Drum hat. Pls do not wear
    """
    n = int(length_s * sr)
    noise = np.random.uniform(-1.0, 1.0, n)
    env = adsr_env_fitted(n, sr, a=0.0005, d=0.03, s=0.0, r=0.02)
    sig = noise * env
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig