#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SnekSynth

A seemingly simple, self-sufficient serpentine sonic synthesizer script.

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



from typing import Any

from numpy import dtype, ndarray

from sneksynth.demo import build_mix
from sneksynth.io import save_wav


def run_demo() -> None:
    """
    Run the demo and save the output to a WAV file.
    """
    mix: ndarray[tuple[Any, ...], dtype[Any]] = build_mix()
    save_wav("demo.wav", mix, 44100)

if __name__ == "__main__":
    run_demo()
