# ===== ADSR Envelope =====
import numpy as np


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