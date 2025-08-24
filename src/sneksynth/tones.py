# ===== Tone Map =====
NATS = "CDEFGAB"
MASK = f"{110:07b}"  # 1101110
STEPS = zip(NATS, MASK)
TONES = dict(
    zip([a for n, b in STEPS for a in (n, (n + "#") * int(b)) if a], range(12))
)
TONES.update(
    {
        f"{nx}B": TONES[f"{n}#"]
        for n, nx, b in zip(NATS, NATS[1:] + NATS[:1], MASK)
        if b == "1"
    }
)
TONES = dict(sorted(list(TONES.items())))
# Weird flex but ok


def note_to_freq(name: str) -> float:
    """
    Convert conventional pitch notation (C0=0 indexing) to frequency in Hz.
    Equal temperament with A4 = 440 Hz.
    """
    a4_index = 9 + 12 * 4  # index 57
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
    semitones_from_a4 = idx - a4_index
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))
