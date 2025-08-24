#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SnekSynth

A seemingly simple, self-sufficient serpentine sonic synthesizer script.

Monolithic demo script for next version features

Output: sneksynth-65-demo.wav
"""

import math
import random
import wave
from typing import Any, Dict, List, Tuple

import numpy as np

# ==== Session ====
SAMPLE_RATE = 44100
BPM = 160
BEATS_PER_BAR = 3
NOTE_DIVISION = 2  # eighths (2 steps/beat)
SECONDS_PER_BEAT = 60.0 / BPM
SECONDS_PER_STEP = SECONDS_PER_BEAT / NOTE_DIVISION
OUT_PATH = "sneksynth-65-demo.wav"
RNG_SEED = 1337

# ==== Note parsing (mask trick) ====
NATS = "CDEFGAB"
MASK = f"{110:07b}"
STEPS = zip(NATS, MASK)
TONES = dict(zip([a for n, b in STEPS for a in (n, (n + "#") * int(b)) if a], range(12)))
TONES.update({f"{nx}B": TONES[f"{n}#"] for n, nx, b in zip(NATS, NATS[1:] + NATS[:1], MASK) if b == "1"})
TONES = dict(sorted(list(TONES.items())))

def note_to_freq(name: str) -> float:
    """
    Convert pitch notation to frequency (A4 = 440 Hz).
    """
    a4_index = 9 + 12 * 4
    name = name.strip().upper()
    if len(name) < 2:
        raise ValueError(f"Bad note name: {name}")
    if name[1] in ["#", "B"]:
        key = name[:2]
        octave = int(name[2:])
    else:
        key = name[0]
        octave = int(name[1:])
    if key not in TONES:
        raise ValueError(f"Bad note key: {key}")
    idx = TONES[key] + 12 * octave
    return 440.0 * (2.0 ** ((idx - a4_index) / 12.0))

def shift_semitones(freq: float, semitones: int) -> float:
    """
    Shift a given frequency by a specified number of semitones.

    This function calculates the new frequency obtained by shifting the input
    frequency up or down by the specified number of semitones. A semitone is the 
    smallest musical interval commonly used in Western music, and shifting by 
    semitones corresponds to moving up or down the chromatic scale.

    Args:
        freq (float): The original frequency in Hertz (Hz).
        semitones (int): The number of semitones to shift. Positive values shift 
            the frequency up, while negative values shift it down.

    Returns:
        float: The frequency in Hertz (Hz) after the semitone shift.
    """
    return freq * (2.0 ** (semitones / 12.0))

# ==== Envelopes ====
def adsr_env_fitted(length_samples: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """
    Generates a fitted linear ADSR (Attack, Decay, Sustain, Release) envelope.
    This function creates an ADSR envelope for a given note length and sample rate.
    The envelope is adjusted proportionally if the sum of the attack, decay, and 
    release times exceeds the total note length. The sustain level is maintained 
    for the remaining duration of the note.

    Args:
        length_samples (int): The total length of the envelope in samples.
        sr (int): The sample rate in Hz.
        a (float): Attack time in seconds.
        d (float): Decay time in seconds.
        s (float): Sustain level (0.0 to 1.0).
        r (float): Release time in seconds.

    Returns:
        np.ndarray: A NumPy array representing the ADSR envelope.

    Notes:
        - If the sum of attack, decay, and release times exceeds the note length, 
          the segments are proportionally scaled down to fit within the note length.
        - The sustain level is clamped to the remaining duration after attack, 
          decay, and release segments are allocated.
    """
    A = max(1, int(round(a * sr)))
    D = max(1, int(round(d * sr)))
    R = max(1, int(round(r * sr)))
    total = A + D + R
    if total > length_samples:
        scale = length_samples / float(total)
        A = max(1, int(round(A * scale)))
        D = max(1, int(round(D * scale)))
        R = max(1, int(round(R * scale)))
        diff = length_samples - (A + D + R)
        if diff > 0:
            add_d = min(diff, 2)
            D += add_d
            diff -= add_d
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
    S_len = max(0, length_samples - (A + D + R))
    env = np.zeros(length_samples, dtype=np.float64)
    env[:A] = np.linspace(0.0, 1.0, A, endpoint=False)
    env[A:A + D] = np.linspace(1.0, s, D, endpoint=False)
    env[A + D:A + D + S_len] = s
    env[A + D + S_len:] = np.linspace(s, 0.0, R, endpoint=False)
    return env

# ===== Wibbly Wobbly Bois =====
def osc_sine(freq: float, length_s: float, sr: int) -> np.ndarray:
    n = int(length_s * sr)
    t: np.ndarray[Tuple[Any], np.dtype[np.float64]] = np.linspace(0.0, length_s, n, endpoint=False)
    return np.sin(2.0 * math.pi * freq * t)

def osc_square(freq: float, length_s: float, sr: int) -> np.ndarray:
    s: np.ndarray[Tuple[Any], np.dtype[Any]] = osc_sine(freq, length_s, sr)
    return np.sign(s)

def osc_triangle(freq: float, length_s: float, sr: int, partials: int = 15) -> np.ndarray:
    """
    Band-limited-ish triangle via additive synthesis of odd harmonics (1/k^2 amplitude).
    """
    n = int(length_s * sr)
    t: np.ndarray[Tuple[Any], np.dtype[np.float64]] = np.linspace(0.0, length_s, n, endpoint=False)
    y: np.ndarray[Tuple[Any], np.dtype[np.float64]] = np.zeros_like(t)
    for k in range(1, 2 * partials, 2):
        y += ((-1) ** ((k - 1) // 2)) * (1.0 / (k * k)) * np.sin(2.0 * math.pi * k * freq * t)
    y = y * (8.0 / (math.pi ** 2))
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def detuned_saws(freq: float, length_s: float, sr: int, detune_cents: float = 7.0, partials: int = 12) -> np.ndarray:
    """
    Detuned twin band-limited saws, summed and normalized.
    """
    n = int(length_s * sr)
    t: np.ndarray[Tuple[Any], np.dtype[np.float64]] = np.linspace(0.0, length_s, n, endpoint=False)
    cents = detune_cents / 1200.0
    f1 = freq * (2.0 ** cents)
    f2 = freq * (2.0 ** -cents)
    def bl_saw(f: float) -> np.ndarray:
        wave = np.zeros_like(t)
        for k in range(1, partials + 1):
            wave += (1.0 / k) * np.sin(2.0 * math.pi * k * f * t)
        peak = np.max(np.abs(wave))
        return wave / peak if peak > 0.0 else wave
    mix = 0.5 * (bl_saw(f1) + bl_saw(f2))
    peak = np.max(np.abs(mix))
    return mix / peak if peak > 0.0 else mix

# ==== Drums ====
def drum_kick(length_s: float, sr: int, f_start: float = 100.0, f_end: float = 40.0) -> np.ndarray:
    """
    Pitch sweep sine + smol click.
    """
    n = int(length_s * sr)
    k = (f_end / f_start) ** (1.0 / max(1, n))
    freqs = f_start * (k ** np.arange(n))
    phase = 2.0 * math.pi * np.cumsum(freqs) / sr
    body = np.sin(phase) * adsr_env_fitted(n, sr, a=0.001, d=0.06, s=0.3, r=0.08)
    click: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.zeros_like(body)
    click_len = max(4, int(0.002 * sr))
    click[:click_len] = 1.0
    click *= adsr_env_fitted(n, sr, a=0.0005, d=0.01, s=0.0, r=0.02)
    sig = 0.95 * body + 0.15 * click
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig

def drum_snare(length_s: float, sr: int) -> np.ndarray:
    """
    Snare = short hiss + low sine thunk.
    """
    n = int(length_s * sr)
    noise = np.random.uniform(-1.0, 1.0, n) * adsr_env_fitted(n, sr, a=0.0005, d=0.12, s=0.0, r=0.08)
    thunk = osc_sine(180.0, length_s, sr) * adsr_env_fitted(n, sr, a=0.001, d=0.05, s=0.0, r=0.05)
    sig = 0.8 * noise + 0.25 * thunk
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig

def drum_hat(length_s: float, sr: int, openish: bool = False) -> np.ndarray:
    """
    Drum hat. Pls do not wear
    """
    n = int(length_s * sr)
    if openish:
        env: np.ndarray[Tuple[Any], np.dtype[Any]] = adsr_env_fitted(n, sr, a=0.0005, d=0.08, s=0.0, r=0.05)
    else:
        env = adsr_env_fitted(n, sr, a=0.0005, d=0.03, s=0.0, r=0.02)
    sig = np.random.uniform(-1.0, 1.0, n) * env
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig

# ==== FX ====
def fx_bass_fall(length_s: float, sr: int, f_start: float = 90.0, f_end: float = 24.0,
                 vibrato_hz: float = 5.0, vibrato_depth_cents: float = 10.0) -> np.ndarray:
    """Downward pitch sweep with a lil vibrato."""
    n = int(length_s * sr)
    t: np.ndarray[Tuple[Any], np.dtype[np.float64]] = np.linspace(0.0, length_s, n, endpoint=False)
    k = (f_end / f_start) ** (1.0 / max(1, n))
    base_freq = f_start * (k ** np.arange(n))
    vib = np.sin(2.0 * math.pi * vibrato_hz * t) * (2.0 ** (vibrato_depth_cents / 1200.0) - 1.0)
    freqs = base_freq * (1.0 + vib)
    phase = 2.0 * math.pi * np.cumsum(freqs) / sr
    tone: np.ndarray[Tuple[Any], np.dtype[Any]] = np.sin(phase)
    env: np.ndarray[Tuple[Any], np.dtype[Any]] = adsr_env_fitted(n, sr, a=0.01, d=0.25, s=0.5, r=0.3)
    sig = tone * env
    return sig / max(1e-12, np.max(np.abs(sig)))

# ==== Scheduling / Mixing ====
def place(buffer: np.ndarray, start_idx: int, signal: np.ndarray, gain: float) -> None:
    """
    Mixes a signal into a buffer at a specified starting index with a given gain.

    This function modifies the `buffer` in-place by adding a scaled version of the
    `signal` starting at `start_idx`. If the signal extends beyond the end of the
    buffer, it is truncated to fit.

    Args:
        buffer (np.ndarray): The target buffer where the signal will be placed.
            Must be a 1D NumPy array.
        start_idx (int): The starting index in the buffer where the signal will
            be added. If `start_idx` is out of bounds or negative, the function
            will handle it gracefully.
        signal (np.ndarray): The signal to be added to the buffer. Must be a 1D
            NumPy array.
        gain (float): The scaling factor applied to the signal before adding it
            to the buffer.

    Returns:
        None: This function modifies the `buffer` in-place and does not return
        any value.
    """
    end_idx = min(start_idx + signal.shape[0], buffer.shape[0])
    seg_len = end_idx - start_idx
    if seg_len <= 0:
        return
    buffer[start_idx:end_idx] += gain * signal[:seg_len]

def schedule_note(buffer: np.ndarray, start_step: int, steps_len: int, osc_fn, freq: float,
                        env_args: Tuple[float, float, float, float], gain: float) -> None:
    """
    Schedules a note to be placed into an audio buffer.

    Args:
        buffer (np.ndarray): The audio buffer where the note will be placed.
        start_step (int): The starting step index for the note in the sequence.
        steps_len (int): The length of the note in steps.
        osc_fn (Callable[[float, float, int], np.ndarray]): The oscillator function to generate the waveform.
            It should take frequency, duration, and sample rate as arguments and return a waveform array.
        freq (float): The frequency of the note in Hz.
        env_args (Tuple[float, float, float, float]): The ADSR envelope parameters as a tuple:
            (attack_time, decay_time, sustain_level, release_time).
        gain (float): The gain multiplier to apply to the note's amplitude.

    Returns:
        None: This function modifies the `buffer` in place.
    """
    start_idx = int(start_step * SECONDS_PER_STEP * SAMPLE_RATE)
    length_s = steps_len * SECONDS_PER_STEP
    n = int(length_s * SAMPLE_RATE)
    if n <= 0:
        return
    raw = osc_fn(freq, length_s, SAMPLE_RATE)
    env: np.ndarray[Tuple[Any], np.dtype[Any]] = adsr_env_fitted(n, SAMPLE_RATE, *env_args)
    place(buffer, start_idx, raw * env, gain)

def schedule_drum(buffer: np.ndarray, start_step: int, steps_len: int, drum_fn, gain: float, **kwargs) -> None:
    """
    Schedules a drum sound to be placed into an audio buffer.

    Args:
        buffer (np.ndarray): The audio buffer where the drum sound will be placed.
        start_step (int): The starting step (time position) in the sequence for the drum sound.
        steps_len (int): The length of the drum sound in steps.
        drum_fn (callable): A function that generates the drum sound signal. It should accept
            the duration (in seconds) and the sample rate as arguments, along with any additional
            keyword arguments.
        gain (float): The gain (volume multiplier) to apply to the drum sound.
        **kwargs: Additional keyword arguments to pass to the `drum_fn`.

    Returns:
        None
    """
    start_idx = int(start_step * SECONDS_PER_STEP * SAMPLE_RATE)
    length_s = steps_len * SECONDS_PER_STEP
    sig = drum_fn(length_s, SAMPLE_RATE, **kwargs)
    place(buffer, start_idx, sig, gain)

# ==== Helpers: Automation Curves ====
def lerp(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """
    Linearly interpolates between two values, `a` and `b`, based on the parameter `t`.

    Parameters:
        a (float): The starting value.
        b (float): The ending value.
        t (np.ndarray): A NumPy array of interpolation factors, where each value should be in the range [0, 1].

    Returns:
        np.ndarray: A NumPy array containing the interpolated values, calculated as `a + (b - a) * t`.
    """
    return a + (b - a) * t

def exp_curve(a: float, b: float, t: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Generates an exponential-like S-curve between two values, `a` and `b`, based on the input array `t`.

    The curve is controlled by the steepness parameter `k`, where `k > 1` results in a steeper curve.

    Parameters:
        a (float): The starting value of the curve.
        b (float): The ending value of the curve.
        t (np.ndarray): A NumPy array of values between 0 and 1 representing the interpolation parameter.
        k (float, optional): The steepness of the curve. Default is 3.0. Higher values result in a steeper curve.

    Returns:
        np.ndarray: A NumPy array of values representing the exponential-like S-curve between `a` and `b`.

    Notes:
        - The input `t` should contain values in the range [0, 1] for the function to behave as expected.
        - The parameter `k` must be greater than 1 to ensure a valid S-curve shape.
    """
    s = (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k))
    return a + (b - a) * s

def segment_envelope(total_samples: int, segments: List[Tuple[int, float, float, str]]) -> np.ndarray:
    """
    Generates a piecewise envelope for a given number of samples based on specified segments.

    This function creates an envelope array where the amplitude values transition between
    specified start and end values over defined segments. The transition can be either
    linear or exponential.

    Args:
        total_samples (int): The total number of samples in the envelope.
        segments (List[Tuple[int, float, float, str]]): A list of segments defining the envelope.
            Each segment is a tuple containing:
            - end_sample (int): The sample index where the segment ends.
            - start_val (float): The starting value of the segment.
            - end_val (float): The ending value of the segment.
            - mode (str): The interpolation mode, either "lin" for linear or "exp" for exponential.

    Returns:
        np.ndarray: A NumPy array representing the generated envelope, with a length of `total_samples`.

    Notes:
        - If the segments do not cover the entire range of `total_samples`, the remaining
          samples are filled with the last segment's end value.
        - The function ensures that each segment starts immediately after the previous one,
          with no gaps in the envelope.
    """
    
    env: np.ndarray[Tuple[int], np.dtype[np.float64]] = np.zeros(total_samples, dtype=np.float64)
    prev_end = 0
    prev_val = segments[0][1] if segments else 1.0
    for end_idx, v0, v1, mode in segments:
        end_idx = max(end_idx, prev_end + 1)
        t: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.linspace(0.0, 1.0, end_idx - prev_end, endpoint=False)
        seg: np.ndarray[Tuple[Any], np.dtype[Any]] = exp_curve(v0, v1, t) if mode == "exp" else lerp(v0, v1, t)
        env[prev_end:end_idx] = seg
        prev_end = end_idx
        prev_val = v1
    if prev_end < total_samples:
        env[prev_end:] = prev_val
    return env

# ==== Musical content & arrangement ====
def bars_waltz_spec(total_bars: int) -> List[Tuple[str, str, str, str]]:
    """
    Return list of bars w/ light variation.
    """
    base = [
        ("D4", "A4", "A4"),
        ("A3", "E4", "E4"),
        ("F3", "C4", "C4"),
        ("G3", "D4", "D4")
    ]
    bars: List[Tuple[str, str, str, str]] = []
    for i in range(total_bars):
        b, s1, s2 = base[i % 4]
        quality = "maj"
        if (i + 1) % 12 == 0:
            quality = "min"
        if RNG_SEED is not None: # type: ignore
            random.seed(RNG_SEED + i * 17)
        if random.random() < 0.15:
            s2 = _transpose_name(s2, 2)  # cheeky whole-step lift
        if (i // 8) % 2 == 1:
            b = _transpose_name(b, 12)
            s1 = _transpose_name(s1, 12)
            s2 = _transpose_name(s2, 12)
        bars.append((b, s1, s2, quality))
    return bars

def _transpose_name(name: str, semitones: int) -> str:
    """
    Transpose a musical note name by a specified number of semitones.

    This function takes a note name (e.g., "C4", "G#3", "Bb5") and transposes it
    by the given number of semitones. The output note name will prefer sharps 
    (e.g., "G#" instead of "Ab") when possible.

    Args:
        name (str): The name of the note to transpose. It should be in the format
            of a note letter (A-G), optionally followed by a sharp (#) or flat (b),
            and ending with an octave number (e.g., "C4", "G#3", "Bb5").
        semitones (int): The number of semitones to transpose the note. Positive
            values transpose up, and negative values transpose down.

    Returns:
        str: The transposed note name, formatted similarly to the input.

    Raises:
        ValueError: If the input note name is not in a valid format or cannot be
            parsed.

    """
    name = name.strip().upper()
    if name[1] in ["#", "B"]:
        key = name[:2]
        octave = int(name[2:])
    else:
        key = name[0]
        octave = int(name[1:])
    base_idx = TONES[key] + 12 * octave
    new_idx = base_idx + semitones
    new_oct = new_idx // 12
    within = new_idx % 12
    sharp_keys = [k for k, v in TONES.items() if len(k) == 2 and k.endswith("#") and v == within]
    nat_keys   = [k for k, v in TONES.items() if len(k) == 1 and v == within]
    key_out = sharp_keys[0] if sharp_keys else (nat_keys[0] if nat_keys else "C")
    return f"{key_out}{new_oct}"

ARRANGEMENT: Dict[int, Dict[str, bool]] = {
    0:  {"bass": True,  "stabs": False, "pad": False, "arp": False, "kick": True,  "snare": False, "hat": True},
    8:  {"bass": True,  "stabs": True,  "pad": False, "arp": False, "kick": True,  "snare": False, "hat": True},
    16: {"bass": True,  "stabs": False, "pad": True,  "arp": False, "kick": True,  "snare": False, "hat": True},
    24: {"bass": True,  "stabs": True,  "pad": True,  "arp": True,  "kick": True,  "snare": True,  "hat": True},
    32: {"bass": True,  "stabs": False, "pad": True,  "arp": False, "kick": True,  "snare": False, "hat": True},
    40: {"bass": True,  "stabs": True,  "pad": True,  "arp": True,  "kick": True,  "snare": True,  "hat": True},
    48: {"bass": True,  "stabs": True,  "pad": True,  "arp": True,  "kick": True,  "snare": True,  "hat": True},
    56: {"bass": True,  "stabs": True,  "pad": True,  "arp": True,  "kick": True,  "snare": True,  "hat": True},
}

def flags_for_bar(bar_idx: int) -> Dict[str, bool]:
    anchors = [k for k in ARRANGEMENT.keys() if k <= bar_idx]
    anchor = max(anchors) if anchors else 0
    return ARRANGEMENT[anchor]

# ==== Automation ====
def automation_plan(total_samples: int, steps_per_bar: int, total_bars_for_env: int) -> Dict[str, np.ndarray]:
    """Granular volume automation controller for the stems."""
    def bars_to_samples(bars: float) -> int:
        steps = int(bars * steps_per_bar)
        return int(steps * SECONDS_PER_STEP * SAMPLE_RATE)

    last_bar = total_bars_for_env  # includes the extra finale bar
    envs: Dict[str, np.ndarray] = {}

    # Master: intro fade, mid lift, big crescendo towards last 8, gentle settle
    envs["master"] = segment_envelope(total_samples, [
        (bars_to_samples(4),   0.60, 0.90, "exp"),
        (bars_to_samples(24),  0.90, 1.00, "lin"),
        (bars_to_samples(56),  1.00, 1.12, "exp"),
        (bars_to_samples(last_bar), 1.12, 1.05, "lin"),
        (total_samples,         1.05, 1.02, "lin"),
    ])
    envs["pad"] = segment_envelope(total_samples, [
        (bars_to_samples(16), 0.0, 0.65, "exp"),
        (bars_to_samples(32), 0.65, 0.55, "lin"),
        (bars_to_samples(56), 0.55, 0.80, "exp"),
        (bars_to_samples(last_bar), 0.80, 0.95, "exp"),
        (total_samples, 0.95, 0.90, "lin"),
    ])
    envs["stabs"] = segment_envelope(total_samples, [
        (bars_to_samples(8),   0.0, 0.70, "exp"),
        (bars_to_samples(24),  0.70, 0.85, "lin"),
        (bars_to_samples(48),  0.85, 0.75, "lin"),
        (bars_to_samples(last_bar), 0.75, 0.65, "lin"),
        (total_samples, 0.65, 0.60, "lin"),
    ])
    envs["bass"] = segment_envelope(total_samples, [
        (bars_to_samples(32), 0.80, 0.85, "lin"),
        (bars_to_samples(56), 0.85, 0.95, "exp"),
        (bars_to_samples(last_bar), 0.95, 0.80, "lin"),
        (total_samples, 0.80, 0.75, "lin"),
    ])
    envs["arp"] = segment_envelope(total_samples, [
        (bars_to_samples(24), 0.0, 0.0, "lin"),
        (bars_to_samples(40), 0.0, 0.55, "exp"),
        (bars_to_samples(56), 0.55, 0.75, "lin"),
        (bars_to_samples(last_bar), 0.75, 0.60, "lin"),
        (total_samples, 0.60, 0.50, "lin"),
    ])
    envs["drums"] = segment_envelope(total_samples, [
        (bars_to_samples(8),   0.40, 0.70, "exp"),
        (bars_to_samples(32),  0.70, 0.90, "lin"),
        (bars_to_samples(56),  0.90, 1.00, "exp"),
        (bars_to_samples(last_bar), 1.00, 0.70, "lin"),
        (total_samples, 0.70, 0.50, "lin"),
    ])
    return envs

# ==== Build & Render ====
def build_mix(main_bars: int = 64, add_final_bar: bool = True) -> np.ndarray:
    """
    This is a lil song I like to call "I Code When I'm Stressed"
    32 bars, 3/4 time, 160bpm
    """
    random.seed(RNG_SEED)
    bars_spec: List[Tuple[str, str, str, str]] = bars_waltz_spec(main_bars)
    steps_per_bar = BEATS_PER_BAR * NOTE_DIVISION

    # Ring-out space for final bar
    extra_end_steps = 6
    finale_steps = steps_per_bar if add_final_bar else 0

    total_steps = main_bars * steps_per_bar + extra_end_steps + finale_steps
    total_samples = int(total_steps * SECONDS_PER_STEP * SAMPLE_RATE)

    stems: Dict[str, np.ndarray[Tuple[int], np.dtype[np.float64]]] = {
        "bass":  np.zeros(total_samples, dtype=np.float64),
        "stabs": np.zeros(total_samples, dtype=np.float64),
        "pad":   np.zeros(total_samples, dtype=np.float64),
        "arp":   np.zeros(total_samples, dtype=np.float64),
        "drums": np.zeros(total_samples, dtype=np.float64),
    }

    # gains (pre-automation)
    gains = {
        "bass": 0.42,
        "stab": 0.34,
        "pad": 0.22,
        "arp": 0.18,
        "kick": 0.85,
        "snare": 0.65,
        "hat": 0.20,
    }

    # ---- Main 64 bars ----
    for bar_index, (bass_note, stab1, stab2, quality) in enumerate(bars_spec):
        flags: Dict[str, bool] = flags_for_bar(bar_index)
        bar_start_step = bar_index * steps_per_bar

        if flags["bass"]:
            f = note_to_freq(bass_note)
            schedule_note(stems["bass"], bar_start_step + 0, 2, osc_sine, f, (0.003, 0.08, 0.55, 0.10), gains["bass"])

        if flags["stabs"]:
            f1 = note_to_freq(stab1)
            f2 = note_to_freq(stab2)
            jump = 12 if (bar_index % 16 == 7) else 0
            schedule_note(stems["stabs"], bar_start_step + 2, 1,
                          lambda ff, _length, s: detuned_saws(ff, _length, s, detune_cents=9.0, partials=14),
                          f1, (0.002, 0.05, 0.35, 0.06), gains["stab"])
            schedule_note(stems["stabs"], bar_start_step + 4, 1,
                          lambda ff, _length, s: detuned_saws(ff, _length, s, detune_cents=9.0, partials=14),
                          f2 * (2.0 ** (jump / 12.0)), (0.002, 0.05, 0.35, 0.06), gains["stab"])

        if flags["pad"]:
            root_f = note_to_freq(bass_note)
            schedule_note(stems["pad"], bar_start_step + 0, 6,
                          lambda ff, _length, s: osc_triangle(ff, _length, s, partials=12),
                          root_f, (0.02, 0.25, 0.5, 0.25), gains["pad"])

        # Arp patterns
        if flags["arp"]:
            root_f = note_to_freq(bass_note)
            third = shift_semitones(root_f, 3 if quality == "min" else 4)
            fifth = shift_semitones(root_f, 7)
            octave = shift_semitones(root_f, 12)
            mode = (bar_index // 8) % 4
            if mode == 0:
                pts = [root_f, third, fifth, octave]
            elif mode == 1:
                pts = [root_f, fifth, third, root_f]
            elif mode == 2:
                pts = [third, octave, fifth, third]
            else:
                pts = [root_f, third, fifth, octave]
            env_a = 0.001 if mode == 3 else 0.003
            for k, p in enumerate(pts):
                schedule_note(stems["arp"], bar_start_step + 2 + k, 1, osc_square, p,
                              (env_a, 0.03, 0.2, 0.04), gains["arp"])

        if flags["kick"]:
            schedule_drum(stems["drums"], bar_start_step + 0, 1, drum_kick, gains["kick"])
            if bar_index >= 40 and (bar_index % 4 in [1, 3]):
                schedule_drum(stems["drums"], bar_start_step + 2, 1, drum_kick, 0.55 * gains["kick"])

        if flags["snare"]:
            schedule_drum(stems["drums"], bar_start_step + 4, 1, drum_snare, gains["snare"])
            if bar_index % 16 == 15:
                schedule_drum(stems["drums"], bar_start_step + 4, 1, drum_snare, 0.35 * gains["snare"])
                schedule_drum(stems["drums"], bar_start_step + 5, 1, drum_snare, 0.50 * gains["snare"])

        if flags["hat"]:
            openish = (bar_index % 8 in [3, 7])
            schedule_drum(stems["drums"], bar_start_step + 2, 1, drum_hat, gains["hat"], openish=openish)
            schedule_drum(stems["drums"], bar_start_step + 4, 1, drum_hat, 0.8 * gains["hat"], openish=False)
            if bar_index >= 8:
                schedule_drum(stems["drums"], bar_start_step + 5, 1, drum_hat, 0.6 * gains["hat"], openish=False)

        # Pre-ending
        if bar_index == 62:
            start = int((bar_start_step + 0) * SECONDS_PER_STEP * SAMPLE_RATE)
            end   = int((bar_start_step + 1) * SECONDS_PER_STEP * SAMPLE_RATE)
            for stem in stems.values():
                stem[start:end] *= 0.0

    # ---- Ending ----
    last_bar_index = len(bars_spec) - 1  # 63
    last_bar_start_step = last_bar_index * steps_per_bar
    fall_start_step = last_bar_start_step + 3
    fall_steps = 3
    fall_sig = fx_bass_fall(fall_steps * SECONDS_PER_STEP, SAMPLE_RATE)
    place(stems["bass"], int(fall_start_step * SECONDS_PER_STEP * SAMPLE_RATE), fall_sig, 0.40)
    minor_ending = False
    bass_note, _, _, _q = bars_spec[-1]
    root = note_to_freq(bass_note)
    third = shift_semitones(root, 3 if minor_ending else 4)
    fifth = shift_semitones(root, 7)

    # Pad triad + octave
    for f, g in [(root, 0.30), (third, 0.28), (fifth, 0.28)]:
        schedule_note(stems["pad"], last_bar_start_step + 0, 6,
                      lambda ff, _length, s: osc_triangle(ff, _length, s, partials=14),
                      f, (0.03, 0.40, 0.60, 1.20), g)
    schedule_note(stems["stabs"], last_bar_start_step + 0, 6,
                  lambda ff, _length, s: detuned_saws(ff, _length, s, detune_cents=6.0, partials=16),
                  shift_semitones(root, 12), (0.02, 0.25, 0.5, 1.2), 0.22)

    if add_final_bar:
        finale_bar_index = main_bars  # 64
        finale_start_step = finale_bar_index * steps_per_bar
        d3 = note_to_freq("D3")
        a3 = note_to_freq("A3")
        d4 = note_to_freq("D4")
        fs4 = note_to_freq("F#4")
        a4 = note_to_freq("A4")
        d5 = note_to_freq("D5")
        pad_env  = (0.04, 0.45, 0.70, 1.80)
        saw_env  = (0.03, 0.35, 0.60, 1.60)
        pad_gain_final = 0.32
        saw_gain_final = 0.22

        for f in [d3, a3, d4, fs4, a4]:
            schedule_note(stems["pad"], finale_start_step + 0, 6,
                          lambda ff, _length, s: osc_triangle(ff, _length, s, partials=16),
                          f, pad_env, pad_gain_final)
        schedule_note(stems["stabs"], finale_start_step + 0, 6,
                      lambda ff, _length, s: detuned_saws(ff, _length, s, detune_cents=5.5, partials=18),
                      d5, saw_env, saw_gain_final)

    # ---- Automation & Sum ----
    total_bars_for_env = main_bars + (1 if add_final_bar else 0)
    envs: Dict[str, np.ndarray[Tuple, np.dtype[np.float64]]] = automation_plan(total_samples, steps_per_bar, total_bars_for_env)
    master_env: np.ndarray[Any, np.dtype[np.float64]] = envs["master"]
    summed: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(total_samples, dtype=np.float64)
    for name, buf in stems.items():
        env: np.ndarray[Any, np.dtype[np.float64]] = envs.get(name, np.ones_like(master_env))
        summed += buf * env
    summed *= master_env

    peak = np.max(np.abs(summed))
    if peak > 0.0:
        summed = summed / peak * 0.98
    return summed

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Save mono float64 buffer as 16-bit PCM WAV."""
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        pcm: np.ndarray[Any, np.dtype[np.signedinteger]] = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        f.writeframes(pcm.tobytes())

if __name__ == "__main__":
    mix: np.ndarray[Tuple[Any], np.dtype[Any]] = build_mix(main_bars=64, add_final_bar=True)
    save_wav(OUT_PATH, mix, SAMPLE_RATE)
