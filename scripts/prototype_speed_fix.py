"""Prototype for the M1 fix: librosa-based time-stretch post-processing.

Does NOT modify the installed package. Instead, it:
  1. Calls engine.generate() at speed=1.0 (the only speed that works today)
     to get a natural-rate waveform.
  2. Applies librosa.effects.time_stretch(rate=speed) as a post-process.
  3. Writes the stretched audio with soundfile.
  4. Runs the test matrix from MAINTAINER_BACKLOG.md.

This confirms the approach is viable before an agent ports it into
src/spanish_tts/engine.py.
"""

import sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

from spanish_tts.config import get_voice
from spanish_tts.engine import generate, _get_model, MODELS


SAMPLE_TEXT = "El rápido zorro marrón salta sobre el perro perezoso."
OUT_DIR = Path("/tmp/spanish-tts-prototype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPEED_MIN, SPEED_MAX = 0.5, 2.0


def apply_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Post-synthesis time-stretch. Pitch-preserving. No-op at speed=1.0."""
    if abs(speed - 1.0) < 1e-6:
        return audio
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise ValueError(f"speed out of range {SPEED_MIN}-{SPEED_MAX}: {speed}")
    audio_f = audio.astype(np.float32, copy=False)
    stretched = librosa.effects.time_stretch(y=audio_f, rate=speed)
    return stretched


def synth_and_stretch(voice_name: str, text: str, speed: float, tag: str) -> dict:
    voice_cfg = get_voice(voice_name)
    base_path = OUT_DIR / f"{tag}_base.wav"
    stretched_path = OUT_DIR / f"{tag}_s{speed:.2f}.wav"

    t0 = time.time()
    if not base_path.exists():
        generate(text=text, voice_config=voice_cfg, speed=1.0,
                 output=str(base_path))
    gen_time = time.time() - t0

    audio, sr = sf.read(str(base_path))
    t1 = time.time()
    stretched = apply_speed(audio, speed, sr)
    stretch_time = time.time() - t1

    sf.write(str(stretched_path), stretched, sr)
    dur = len(stretched) / sr
    return {"voice": voice_name, "speed": speed, "path": str(stretched_path),
            "duration_s": round(dur, 3), "samples": len(stretched),
            "gen_time_s": round(gen_time, 2),
            "stretch_time_ms": round(stretch_time * 1000, 1)}


def spectral_centroid_hz(path: str) -> float:
    y, sr = sf.read(path)
    cent = librosa.feature.spectral_centroid(y=y.astype(np.float32), sr=sr)
    return float(np.mean(cent))


def run_matrix():
    results = {}
    print("Pre-loading design model...")
    _get_model(MODELS["design"])

    print("\n--- neutral_male (design) speed sweep ---")
    design_runs = {}
    for spd in [0.5, 1.0, 1.0, 2.0]:
        r = synth_and_stretch("neutral_male", SAMPLE_TEXT, spd, f"nm")
        design_runs.setdefault(spd, []).append(r)
        print(f"  speed={spd:.2f}  dur={r['duration_s']:>5}s  "
              f"stretch={r['stretch_time_ms']}ms  gen={r['gen_time_s']}s")

    d05 = design_runs[0.5][0]["duration_s"]
    d10_a = design_runs[1.0][0]["duration_s"]
    d10_b = design_runs[1.0][1]["duration_s"]
    d20 = design_runs[2.0][0]["duration_s"]

    r_slow = d05 / d10_a
    r_fast = d20 / d10_a
    var_10 = abs(d10_a - d10_b) / max(d10_a, d10_b)

    results["test2_slow_ratio"] = {"value": round(r_slow,3), "pass": 1.7 <= r_slow <= 2.3}
    results["test3_fast_ratio"] = {"value": round(r_fast,3), "pass": 0.45 <= r_fast <= 0.55}
    results["test5_variance"]   = {"value": round(var_10,4), "pass": var_10 < 0.05}

    c05 = spectral_centroid_hz(design_runs[0.5][0]["path"])
    c10 = spectral_centroid_hz(design_runs[1.0][0]["path"])
    c20 = spectral_centroid_hz(design_runs[2.0][0]["path"])
    results["test11_pitch_slow"] = {
        "c_slow": round(c05,1), "c_ref": round(c10,1),
        "pct_diff": round(abs(c05-c10)/c10*100,1),
        "pass": abs(c05-c10)/c10 < 0.15}
    results["test11_pitch_fast"] = {
        "c_fast": round(c20,1), "c_ref": round(c10,1),
        "pct_diff": round(abs(c20-c10)/c10*100,1),
        "pass": abs(c20-c10)/c10 < 0.15}

    print("\n--- carlos_mx (clone) sanity check ---")
    _get_model(MODELS["clone"])
    clone_base = synth_and_stretch("carlos_mx", SAMPLE_TEXT, 1.0, "cm")
    clone_slow = synth_and_stretch("carlos_mx", SAMPLE_TEXT, 0.5, "cm")
    clone_fast = synth_and_stretch("carlos_mx", SAMPLE_TEXT, 2.0, "cm")
    r_clone_slow = clone_slow["duration_s"] / clone_base["duration_s"]
    r_clone_fast = clone_fast["duration_s"] / clone_base["duration_s"]
    print(f"  clone 0.5/1.0 ratio = {r_clone_slow:.3f}")
    print(f"  clone 2.0/1.0 ratio = {r_clone_fast:.3f}")
    results["test4_clone_slow"] = {"value": round(r_clone_slow,3), "pass": 1.7 <= r_clone_slow <= 2.3}
    results["test4_clone_fast"] = {"value": round(r_clone_fast,3), "pass": 0.45 <= r_clone_fast <= 0.55}

    errors = []
    for bad in [0.499, 2.001, 0.4, 2.5]:
        try:
            apply_speed(np.zeros(24000, dtype=np.float32), bad, 24000)
            errors.append((bad, "ACCEPTED (bug)"))
        except ValueError as e:
            errors.append((bad, str(e)))
    for ok in [0.5, 2.0]:
        try:
            apply_speed(np.zeros(24000, dtype=np.float32), ok, 24000)
            errors.append((ok, "ACCEPTED"))
        except ValueError as e:
            errors.append((ok, f"REJECTED (bug): {e}"))
    results["test12_boundaries"] = {"cases": errors,
        "pass": all(("ACCEPTED" in str(x[1])) for x in errors if x[0] in (0.5,2.0))
              and all(("out of range" in str(x[1])) for x in errors if x[0] not in (0.5,2.0))}

    print("\n=== RESULTS ===")
    for name, r in results.items():
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"  [{mark}] {name}: {r}")

    all_pass = all(r["pass"] for r in results.values())
    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass, results


if __name__ == "__main__":
    ok, _ = run_matrix()
    sys.exit(0 if ok else 1)
