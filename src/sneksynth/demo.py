# -*- coding: utf-8 -*-
from typing import Any

import numpy as np

from sneksynth import SAMPLE_RATE
from sneksynth.oscilators import (
    detuned_saws,
    drum_hat,
    drum_kick,
    drum_snare,
    osc_sine,
    osc_square,
    osc_triangle,
)
from sneksynth.sequencer import schedule_drum, schedule_note_tone
from sneksynth.tones import note_to_freq

# ===== Global Session Settings =====
BPM = 160
BEATS_PER_BAR = 3
NOTE_DIVISION = 2

SECONDS_PER_BEAT = 60.0 / BPM
SECONDS_PER_STEP = SECONDS_PER_BEAT / NOTE_DIVISION

OUT_PATH = "demo.wav"


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


# ===== Ok let's make a music =====
def build_mix() -> np.ndarray[tuple[Any, ...], np.dtype[np.float64]]:
    """
    This is a lil song I like to call "I Code When I'm Stressed"
    32 bars, 3/4 time, 160bpm
    """
    bars_spec = bars_waltz_spec(repeats=8)
    steps_per_bar = BEATS_PER_BAR * NOTE_DIVISION
    total_steps = len(bars_spec) * steps_per_bar
    total_samples = int(total_steps * SECONDS_PER_STEP * SAMPLE_RATE)
    mix: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(total_samples, dtype=np.float64)

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