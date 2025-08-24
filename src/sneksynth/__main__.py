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


from sneksynth.next import demo_next


def main():
    """
    Main function to run the SnekSynth demo.
    """
    demo_next()

if __name__ == "__main__":
    main()