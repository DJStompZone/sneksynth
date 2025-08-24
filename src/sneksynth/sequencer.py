# ===== Audacity? Never met her =====
import numpy as np

from sneksynth import SAMPLE_RATE
from sneksynth.envelope import adsr_env_fitted


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
    SECONDS_PER_STEP: float = 60 / 160 / 2 # 60s / 160bpm / 2 steps per beat
) -> None:
    """
    Schedule a tonal note on the step grid.

    Parameters:
        buffer (np.ndarray): The audio buffer to modify.
        start_step (int): The step at which to start the note.
        steps_len (int): The duration of the note in steps.
        osc_fn (function): The oscillator function to generate the tone.
        freq (float): The frequency of the note.
        env_args (tuple): Arguments for the ADSR envelope function.
        gain (float): Gain to apply to the signal.
        SECONDS_PER_STEP (float, optional): Duration of one step in seconds. Defaults to 60 / 160 / 2.
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
    buffer: np.ndarray, start_step: int, steps_len: int, drum_fn, gain: float, SECONDS_PER_STEP: float = 60 / 160 / 2
) -> None:
    """
    Schedule a drum sound.

    Parameters:
        buffer (np.ndarray): The audio buffer to modify.
        start_step (int): The step at which to start the drum sound.
        steps_len (int): The duration of the drum sound in steps.
        drum_fn (function): The drum sound function to generate the signal.
        gain (float): Gain to apply to the signal.
        SECONDS_PER_STEP (float, optional): Duration of one step in seconds. Defaults to 60 / 160 / 2.
    """
    start_idx = int(start_step * SECONDS_PER_STEP * SAMPLE_RATE)
    length_s = steps_len * SECONDS_PER_STEP
    sig = drum_fn(length_s, SAMPLE_RATE)
    place(buffer, start_idx, sig, gain)