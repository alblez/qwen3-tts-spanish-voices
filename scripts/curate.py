#!/usr/bin/env python3
"""Curate Spanish voice references from VoxForge Spanish corpus.

Downloads the dataset, filters for high-quality clips, and extracts
the best candidate per speaker grouped by country and gender.

Usage:
    python curate.py browse                   # Show dataset stats
    python curate.py pick --country mexico --gender male --limit 5
    python curate.py export <speaker_id> --name carlos_mx
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
import soundfile as sf


DATASET_ID = "ciempiess/voxforge_spanish"


def _matches_filter(row, country=None, gender=None):
    """Check if a row matches country/gender filters."""
    if country and (row["country"] or "").lower() != country.lower():
        return False
    if gender and (row["gender"] or "").lower() != gender.lower():
        return False
    return True


def _find_speaker_clips(ds, speaker_id, min_duration, max_duration):
    """Find clips for a speaker within duration range."""
    return [
        (i, row) for i, row in enumerate(ds)
        if row["speaker_id"] == speaker_id
        and min_duration <= row["duration"] <= max_duration
    ]


def _score_speakers(speakers):
    """Score and rank speakers by clip count and text quality."""
    scored = []
    for speaker_id, clips in speakers.items():
        n_clips = len(clips)
        avg_dur = sum(c["duration"] for c in clips) / n_clips
        avg_text_len = sum(len(c["text"]) for c in clips) / n_clips
        best = max(clips, key=lambda c: len(c["text"]) * c["duration"])
        scored.append({
            "speaker_id": speaker_id,
            "n_clips": n_clips,
            "avg_duration": avg_dur,
            "avg_text_len": avg_text_len,
            "best_clip": best,
            "country": clips[0]["country"],
            "gender": clips[0]["gender"],
        })
    scored.sort(key=lambda s: (s["n_clips"], s["avg_text_len"]), reverse=True)
    return scored


def _load_dataset(with_audio=True):
    """Load VoxForge Spanish dataset (cached after first download)."""
    from datasets import load_dataset
    click.echo("Loading VoxForge Spanish dataset (3.4 GB, cached after first download)...")
    ds = load_dataset(DATASET_ID, split="train")
    if not with_audio:
        ds = ds.remove_columns(["audio"])
    click.echo(f"Loaded {len(ds)} samples.")
    return ds


def _load_dataset_raw():
    """Load dataset in arrow format for raw audio bytes (bypass torchcodec)."""
    from datasets import load_dataset
    click.echo("Loading dataset with raw audio...")
    ds = load_dataset(DATASET_ID, split="train")
    return ds.with_format("arrow")


def _decode_audio(ds_raw, idx):
    """Decode audio from raw arrow dataset using soundfile."""
    import io
    row = ds_raw[idx]
    audio_bytes = row.column("audio")[0].as_py()["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    return audio_array.astype(np.float32), sr


@click.group()
def cli():
    """Curate Spanish voice references from VoxForge."""
    pass


@cli.command()
def browse():
    """Show dataset statistics by country and gender."""
    ds = _load_dataset(with_audio=False)

    # Collect stats
    by_country = Counter()
    by_gender = Counter()
    by_combo = Counter()
    speakers_by_combo = defaultdict(set)

    for row in ds:
        country = row["country"] or "unknown"
        gender = row["gender"] or "unknown"
        speaker = row["speaker_id"]
        by_country[country] += 1
        by_gender[gender] += 1
        by_combo[(country, gender)] += 1
        speakers_by_combo[(country, gender)].add(speaker)

    click.echo(f"\nTotal samples: {len(ds)}")
    click.echo(f"\n{'COUNTRY':<18} {'GENDER':<10} {'SAMPLES':<10} {'SPEAKERS'}")
    click.echo("-" * 55)
    for (country, gender), count in sorted(by_combo.items()):
        n_speakers = len(speakers_by_combo[(country, gender)])
        click.echo(f"{country:<18} {gender:<10} {count:<10} {n_speakers}")


@cli.command()
@click.option("--country", default=None, help="Filter by country (mexico, spain, argentina, chile).")
@click.option("--gender", default=None, type=click.Choice(["male", "female"]))
@click.option("--min-duration", default=6.0, type=float, help="Minimum clip duration in seconds.")
@click.option("--max-duration", default=12.0, type=float, help="Maximum clip duration in seconds.")
@click.option("--limit", default=10, type=int, help="Number of candidates to show.")
def pick(country, gender, min_duration, max_duration, limit):
    """Find best voice clone candidates matching criteria.

    Ranks speakers by: number of clips in duration range, average duration,
    and text length (proxy for clear, full sentences).
    """
    ds = _load_dataset(with_audio=False)

    # Group samples by speaker
    speakers = defaultdict(list)
    for i, row in enumerate(ds):
        if not _matches_filter(row, country, gender):
            continue
        dur = row["duration"]
        if min_duration <= dur <= max_duration:
            speakers[row["speaker_id"]].append({
                "index": i,
                "duration": dur,
                "text": row["normalized_text"],
                "country": row["country"],
                "gender": row["gender"],
                "audio_id": row["audio_id"],
            })

    if not speakers:
        click.echo("No speakers found matching criteria.")
        return

    scored = _score_speakers(speakers)

    click.echo(f"\nTop {limit} candidates ({country or 'any'}, {gender or 'any'}):")
    click.echo(f"{'SPEAKER':<14} {'COUNTRY':<14} {'CLIPS':<7} {'AVG_DUR':<9} {'BEST_DUR':<10} {'BEST_TEXT'}")
    click.echo("-" * 100)
    for s in scored[:limit]:
        best = s["best_clip"]
        text_preview = best["text"][:50] + "..." if len(best["text"]) > 50 else best["text"]
        click.echo(
            f"{s['speaker_id']:<14} {s['country']:<14} {s['n_clips']:<7} "
            f"{s['avg_duration']:<9.1f} {best['duration']:<10.1f} {text_preview}"
        )

    # Save candidates to JSON for reference
    output_file = Path("candidates.json")
    with open(output_file, "w") as f:
        json.dump(scored[:limit], f, indent=2, ensure_ascii=False)
    click.echo(f"\nSaved details to {output_file}")


@cli.command()
@click.argument("speaker_id")
@click.option("--name", required=True, help="Voice name for the registry (e.g. carlos_mx).")
@click.option("--clip-index", default=None, type=int,
              help="Specific dataset index. If not given, picks the best clip.")
@click.option("--min-duration", default=6.0, type=float)
@click.option("--max-duration", default=12.0, type=float)
def export(speaker_id, name, clip_index, min_duration, max_duration):
    """Export a speaker's best clip as a .wav reference and register it."""
    ds_meta = _load_dataset(with_audio=False)

    if clip_index is not None:
        # Use specific clip
        row = ds_meta[clip_index]
        if row["speaker_id"] != speaker_id:
            click.echo(f"Warning: index {clip_index} belongs to {row['speaker_id']}, not {speaker_id}")
        best_idx = clip_index
        meta = row
    else:
        # Find best clip for this speaker
        candidates = _find_speaker_clips(ds_meta, speaker_id, min_duration, max_duration)

        if not candidates:
            click.echo(f"No suitable clips found for speaker {speaker_id}")
            return

        # Pick longest text within duration range
        best_idx, meta = max(candidates, key=lambda x: len(x[1]["normalized_text"]) * x[1]["duration"])
        click.echo(f"Selected clip index {best_idx}: {meta['duration']:.1f}s")

    # Extract audio via raw bytes
    ds_raw = _load_dataset_raw()
    audio_array, sr = _decode_audio(ds_raw, best_idx)

    # Save to references dir
    from spanish_tts.config import add_voice, get_references_dir

    ref_dir = get_references_dir()
    ref_path = ref_dir / f"{name}.wav"
    sf.write(str(ref_path), audio_array, sr)

    # Register voice
    voice_data = {
        "type": "clone",
        "ref_audio": str(ref_path),
        "ref_text": meta["normalized_text"],
        "accent": meta["country"] or "unknown",
        "gender": meta["gender"] or "unknown",
        "language": "Spanish",
        "source_speaker": speaker_id,
        "source_audio_id": meta["audio_id"],
    }
    add_voice(name, voice_data)

    click.echo(f"\nExported voice '{name}':")
    click.echo(f"  Audio:   {ref_path} ({meta['duration']:.1f}s, {sr}Hz)")
    click.echo(f"  Text:    {meta['normalized_text']}")
    click.echo(f"  Country: {meta['country']}")
    click.echo(f"  Gender:  {meta['gender']}")
    click.echo(f"\nUse: spanish-tts say \"Hola mundo\" --voice {name}")


@cli.command("listen")
@click.argument("speaker_id")
@click.option("--limit", default=3, type=int, help="Number of clips to export for preview.")
@click.option("--min-duration", default=6.0, type=float)
@click.option("--max-duration", default=12.0, type=float)
def listen(speaker_id, limit, min_duration, max_duration):
    """Export a speaker's top clips to /tmp for quick listening with afplay."""
    ds_meta = _load_dataset(with_audio=False)

    # Find candidate indices without loading audio
    candidates = _find_speaker_clips(ds_meta, speaker_id, min_duration, max_duration)

    if not candidates:
        click.echo(f"No suitable clips for {speaker_id}")
        return

    # Sort by text length * duration
    candidates.sort(key=lambda x: len(x[1]["normalized_text"]) * x[1]["duration"], reverse=True)
    candidates = candidates[:limit]

    # Load raw audio bytes (bypass torchcodec)
    ds_full = _load_dataset_raw()

    out_dir = Path(f"/tmp/voxforge_preview/{speaker_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, (idx, meta) in enumerate(candidates):
        audio_array, sr = _decode_audio(ds_full, idx)
        out_path = out_dir / f"{j+1}_{meta['audio_id']}.wav"
        sf.write(str(out_path), audio_array, sr)
        click.echo(f"  [{j+1}] {out_path}")
        click.echo(f"      {meta['duration']:.1f}s: {meta['normalized_text'][:80]}")

    click.echo(f"\nPreview: afplay {out_dir}/1_*.wav")


if __name__ == "__main__":
    cli()
