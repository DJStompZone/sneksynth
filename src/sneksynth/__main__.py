#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SnekSynth

Features:
  - Voices:
      - Bass: sine with longer tail
      - Stabs: detuned, band-limited saw pair
      - Pad: triangle-like additive for glue
      - Arp: square-based pseudo-arpeggiator (optional layer)
      - Drums: kick, snare, hats
  - ADSR envelope

Requirements:
  - Numpy is the only nonstandard dependency

Output:
  demo.wav
"""

import math
import wave

import numpy as np

from sneksynth.tones import note_to_freq

# ===== Global Session Settings =====
SAMPLE_RATE = 44100
BPM = 160
BEATS_PER_BAR = 3
NOTE_DIVISION = 2

SECONDS_PER_BEAT = 60.0 / BPM
SECONDS_PER_STEP = SECONDS_PER_BEAT / NOTE_DIVISION

OUT_PATH = "demo.wav"


# ===== ADSR Envelope =====
def adsr_env_fitted(
    length_samples: int, sr: int, a: float, d: float, s: float, r: float
) -> np.ndarray:
    """
    Generates a fitted linear ADSR envelope.

    If attack+decay+release exceeds the note length, segments are
    proportionally scaled down to keep numpy happy.
    """
    # seconds → samples
    A = max(1, int(round(a * sr)))
    D = max(1, int(round(d * sr)))
    R = max(1, int(round(r * sr)))

    total = A + D + R  # The 4th one just wasn't sustainable :D
    if total > length_samples:
        scale = length_samples / float(total)
        A = max(1, int(round(A * scale)))
        D = max(1, int(round(D * scale)))
        R = max(1, int(round(R * scale)))
        diff = length_samples - (A + D + R)
        # Prioritize adding to decay first, then release, then attack
        if diff > 0:
            add_d = min(diff, 2)
            D += add_d
            diff -= add_d
        # How many times do we have to teach you this lesson, old man?
        if diff > 0:
            R += diff
            diff = 0
        if diff < 0:
            take = min(-diff, max(0, D - 1))
            D -= take
            diff += take
        if diff < 0:
            take = min(-diff, max(0, R - 1))
            R -= take
            diff += take

    # Middle lookin a lil sus
    S_len = max(0, length_samples - (A + D + R))

    env = np.zeros(length_samples, dtype=np.float64)
    # Attack
    env[:A] = np.linspace(0.0, 1.0, A, endpoint=False)
    # Decay
    env[A : A + D] = np.linspace(1.0, s, D, endpoint=False)
    # Sustain
    env[A + D : A + D + S_len] = s
    # Release
    env[A + D + S_len :] = np.linspace(s, 0.0, R, endpoint=False)
    return env  # Cha Cha real smooth


# ===== Wibbly Wobbly Bois =====
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


# ===== Audacity? Never met her =====
def place(buffer: np.ndarray, start_idx: int, signal: np.ndarray, gain: float) -> None:
    end_idx = min(start_idx + signal.shape[0], buffer.shape[0])
    seg_len = end_idx - start_idx
    if seg_len <= 0:
        return
    buffer[start_idx:end_idx] += gain * signal[:seg_len]


def schedule_note_tone(
    buffer: np.ndarray,
    start_step: int,
    steps_len: int,
    osc_fn,
    freq: float,
    env_args: tuple,
    gain: float,
) -> None:
    """
    Schedule a tonal note on the step grid.
    """
    start_idx = int(start_step * SECONDS_PER_STEP * SAMPLE_RATE)
    length_s = steps_len * SECONDS_PER_STEP
    n = int(length_s * SAMPLE_RATE)
    if n <= 0:
        return
    raw = osc_fn(freq, length_s, SAMPLE_RATE)
    env = adsr_env_fitted(n, SAMPLE_RATE, *env_args)
    sig = raw * env
    place(buffer, start_idx, sig, gain)


def schedule_drum(
    buffer: np.ndarray, start_step: int, steps_len: int, drum_fn, gain: float
) -> None:
    """
    Schedule a drum sound.
    """
    start_idx = int(start_step * SECONDS_PER_STEP * SAMPLE_RATE)
    length_s = steps_len * SECONDS_PER_STEP
    sig = drum_fn(length_s, SAMPLE_RATE)
    place(buffer, start_idx, sig, gain)


# ===== Ok let's make a music =====
def bars_waltz_spec(repeats: int) -> list[tuple[str, str, str]]:
    """
    Beep boop boop
    """
    base = [
        ("D4", "A4", "A4"),
        ("A3", "E4", "E4"),
        ("F3", "C4", "C4"),
        ("G3", "D4", "D4"),
    ]
    seq = []
    for _ in range(repeats):
        seq.extend(base)
    return seq


def build_mix() -> np.ndarray:
    """
    This is a lil song I like to call "I Code When I'm Stressed"
    32 bars, 3/4 time, 160bpm
    """
    bars_spec = bars_waltz_spec(repeats=8)
    steps_per_bar = BEATS_PER_BAR * NOTE_DIVISION
    total_steps = len(bars_spec) * steps_per_bar
    total_samples = int(total_steps * SECONDS_PER_STEP * SAMPLE_RATE)
    mix = np.zeros(total_samples, dtype=np.float64)

    # bars 0-3,16-19 bass + hats lite
    # bars 4-7,20-23 + stabs
    # bars 8-11,24-27 + pad + sparse hats
    # bars 12-15,28-31 arp + full drums
    def section_flags(bar_idx: int) -> dict:
        if bar_idx < 4:
            return {
                "bass": True,
                "stabs": False,
                "pad": False,
                "arp": False,
                "kick": True,
                "snare": False,
                "hat": True,
            }
        if bar_idx < 8:
            return {
                "bass": True,
                "stabs": True,
                "pad": False,
                "arp": False,
                "kick": True,
                "snare": False,
                "hat": True,
            }
        if bar_idx < 12:
            return {
                "bass": True,
                "stabs": False,
                "pad": True,
                "arp": False,
                "kick": True,
                "snare": False,
                "hat": True,
            }
        return {
            "bass": True,
            "stabs": True,
            "pad": True,
            "arp": True,
            "kick": True,
            "snare": True,
            "hat": True,
        }

    # Gains
    bass_gain = 0.42
    stab_gain = 0.34
    pad_gain = 0.22
    arp_gain = 0.18
    kick_gain = 0.85
    snare_gain = 0.65
    hat_gain = 0.20

    for bar_index, (bass_note, stab1, stab2) in enumerate(bars_spec):
        flags = section_flags(bar_index)
        bar_start_step = bar_index * steps_per_bar

        if flags["bass"]:
            bass_f = note_to_freq(bass_note)
            schedule_note_tone(
                mix,
                bar_start_step + 0,
                2,
                osc_sine,
                bass_f,
                (0.003, 0.08, 0.55, 0.10),
                bass_gain,
            )

        if flags["stabs"]:
            f1 = note_to_freq(stab1)
            f2 = note_to_freq(stab2)
            schedule_note_tone(
                mix,
                bar_start_step + 2,
                1,
                lambda f, _length, s: detuned_saws(f, _length, s, detune_cents=9.0, partials=14),
                f1,
                (0.002, 0.05, 0.35, 0.06),
                stab_gain,
            )
            schedule_note_tone(
                mix,
                bar_start_step + 4,
                1,
                lambda f, _length, s: detuned_saws(f, _length, s, detune_cents=9.0, partials=14),
                f2,
                (0.002, 0.05, 0.35, 0.06),
                stab_gain,
            )

        if flags["pad"]:
            root_f = note_to_freq(bass_note)
            schedule_note_tone(
                mix,
                bar_start_step + 0,
                6,
                lambda f, _length, s: osc_triangle(f, _length, s, partials=12),
                root_f,
                (0.02, 0.25, 0.5, 0.25),
                pad_gain,
            )

        if flags["arp"]:
            root_f = note_to_freq(bass_note)
            third_f = root_f * (2.0 ** (4 / 12.0))  # +4 semitones
            fifth_f = root_f * (2.0 ** (7 / 12.0))  # +7 semitones
            schedule_note_tone(
                mix,
                bar_start_step + 2,
                1,
                osc_square,
                root_f,
                (0.001, 0.03, 0.2, 0.04),
                arp_gain,
            )
            schedule_note_tone(
                mix,
                bar_start_step + 3,
                1,
                osc_square,
                third_f,
                (0.001, 0.03, 0.2, 0.04),
                arp_gain,
            )
            schedule_note_tone(
                mix,
                bar_start_step + 4,
                1,
                osc_square,
                fifth_f,
                (0.001, 0.03, 0.2, 0.04),
                arp_gain,
            )
            schedule_note_tone(
                mix,
                bar_start_step + 5,
                1,
                osc_square,
                root_f,
                (0.001, 0.03, 0.2, 0.04),
                arp_gain,
            )

        # Drums
        if flags["kick"]:
            schedule_drum(mix, bar_start_step + 0, 1, drum_kick, kick_gain)
            if bar_index >= 12:
                schedule_drum(mix, bar_start_step + 3, 1, drum_kick, 0.55 * kick_gain)

        if flags["snare"]:
            schedule_drum(mix, bar_start_step + 4, 1, drum_snare, snare_gain)

        if flags["hat"]:
            schedule_drum(mix, bar_start_step + 2, 1, drum_hat, hat_gain)
            schedule_drum(mix, bar_start_step + 4, 1, drum_hat, 0.8 * hat_gain)
            if bar_index >= 4:
                schedule_drum(mix, bar_start_step + 5, 1, drum_hat, 0.6 * hat_gain)

        # Bring it around
        if not bar_index % 14:
            start = int((bar_start_step + 0) * SECONDS_PER_STEP * SAMPLE_RATE)
            end = int((bar_start_step + 1) * SECONDS_PER_STEP * SAMPLE_RATE)
            mix[start:end] *= 0.0

    # Protec earholes
    peak = np.max(np.abs(mix))
    if peak > 0.0:
        mix = mix / peak * 0.98
    return mix  # fuck it, ship it


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """
    Export as 16-bit PCM WAV.
    """
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        f.writeframes(pcm.tobytes())


if __name__ == "__main__":
    mix = build_mix()
    save_wav(OUT_PATH, mix, SAMPLE_RATE)
