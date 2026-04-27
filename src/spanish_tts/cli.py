# SPDX-License-Identifier: MIT
"""CLI for spanish-tts."""

import click

from spanish_tts.config import (
    add_voice,
    get_defaults,
    get_references_dir,
    get_voice,
    list_voices,
)
from spanish_tts.engine import SPEED_MAX, SPEED_MIN, generate


@click.group()
@click.version_option(package_name="qwen3-tts-spanish-voices")
def cli():
    """Curated Spanish TTS voices using Qwen3-TTS."""
    pass


@cli.command()
@click.argument("text")
@click.option("--voice", "-v", default="neutral_male", help="Voice name from registry.")
@click.option(
    "--speed",
    "-s",
    type=click.FloatRange(SPEED_MIN, SPEED_MAX),
    default=None,
    help="Speed factor (0.5-2.0).",
)
@click.option("--output", "-o", default=None, help="Output .wav path.")
@click.option("--play", "-p", is_flag=True, help="Auto-play with afplay after generating.")
@click.option("--stream", is_flag=True, default=False, help="Use streaming decode (lower memory).")
def say(text, voice, speed, output, play, stream):
    """Generate speech from text using a registered voice."""
    defaults = get_defaults()
    voice_config = get_voice(voice)

    if voice_config is None:
        available = list(list_voices().keys())
        click.echo(f"Error: Voice '{voice}' not found.", err=True)
        click.echo(f"Available: {', '.join(available)}", err=True)
        raise SystemExit(1)

    effective_speed = speed if speed is not None else defaults.get("speed", 1.0)
    output_dir = defaults.get("output_dir", "~/tts-output/spanish")

    result = generate(
        text=text,
        voice_config=voice_config,
        speed=effective_speed,
        output=output,
        output_dir=output_dir,
        stream=stream,
    )
    # Print path to stdout for piping
    click.echo(result)

    if play:
        import subprocess

        subprocess.run(["/usr/bin/afplay", str(result)])


@cli.command("list")
def list_cmd():
    """List all registered voices."""
    voices = list_voices()
    if not voices:
        click.echo("No voices registered.")
        return

    click.echo(f"{'NAME':<20} {'TYPE':<8} {'GENDER':<8} {'ACCENT/DESC'}")
    click.echo("-" * 70)
    for name, config in voices.items():
        vtype = config.get("type", "?")
        gender = config.get("gender", "?")
        if vtype == "clone":
            desc = config.get("accent", config.get("country", ""))
        else:
            instruct = config.get("instruct", "")
            desc = instruct[:40] + "..." if len(instruct) > 40 else instruct
        click.echo(f"{name:<20} {vtype:<8} {gender:<8} {desc}")


@cli.command("add-ref")
@click.argument("name")
@click.argument("audio_path", type=click.Path(exists=True))
@click.argument("transcript")
@click.option("--accent", default="neutral", help="Accent label (e.g. mexico, spain, argentina).")
@click.option("--gender", type=click.Choice(["male", "female"]), required=True)
@click.option(
    "--license",
    "source_license",
    default="user-supplied-unspecified",
    help="SPDX license of the reference audio (e.g. 'GPL-3.0', 'CC-BY-4.0').",
)
@click.option("--source-url", default=None, help="URL to the original audio source.")
def add_ref(name, audio_path, transcript, accent, gender, source_license, source_url):
    """Register a clone voice from a reference audio file."""
    import shutil
    from pathlib import Path

    # Copy reference to managed location
    ref_dir = get_references_dir()
    src = Path(audio_path)
    dest = ref_dir / f"{name}{src.suffix}"
    shutil.copy2(src, dest)

    voice_data: dict = {
        "type": "clone",
        "ref_audio": str(dest),
        "ref_text": transcript,
        "accent": accent,
        "gender": gender,
        "language": "Spanish",
        "source_license": source_license,
    }
    if source_url is not None:
        voice_data["source_url"] = source_url
    add_voice(name, voice_data)
    click.echo(f"Added clone voice '{name}' from {src.name} ({accent}, {gender})")


@cli.command("add-design")
@click.argument("name")
@click.argument("instruct")
@click.option("--gender", type=click.Choice(["male", "female"]), required=True)
def add_design(name, instruct, gender):
    """Register a designed voice from a description."""
    voice_data = {
        "type": "design",
        "instruct": instruct,
        "gender": gender,
        "language": "Spanish",
    }
    add_voice(name, voice_data)
    click.echo(f"Added design voice '{name}' ({gender})")


@cli.command()
def voices():
    """Show detailed info about all voices (alias for list)."""
    ctx = click.get_current_context()
    ctx.invoke(list_cmd)


@cli.command()
@click.argument("name")
def remove(name):
    """Remove a voice from the registry."""
    from spanish_tts.config import load_voices, save_voices

    data = load_voices()
    voices = data.get("voices", {})
    if name not in voices:
        click.echo(f"Voice '{name}' not found.", err=True)
        raise SystemExit(1)
    del voices[name]
    data["voices"] = voices
    save_voices(data)
    click.echo(f"Removed voice '{name}'")


@cli.command()
@click.argument("text")
@click.option(
    "--output-dir", "-d", default="/tmp/spanish-tts-demo", help="Directory for demo files."
)
@click.option(
    "--speed",
    "-s",
    type=click.FloatRange(SPEED_MIN, SPEED_MAX),
    default=1.0,
    help="Speed factor (0.5-2.0).",
)
def demo(text, output_dir, speed):
    """Generate the same text with ALL registered voices for comparison."""
    from pathlib import Path

    voices = list_voices()
    if not voices:
        click.echo("No voices registered.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(f"Generating '{text[:50]}...' with {len(voices)} voices (speed={speed}) -> {out}/\n")
    for name, config in voices.items():
        try:
            result = generate(
                text=text,
                voice_config=config,
                speed=speed,
                output=str(out / f"{name}.wav"),
            )
            click.echo(f"  OK  {name:<20} -> {result}")
        except Exception as e:
            click.echo(f"  FAIL {name:<20} -> {e}", err=True)

    click.echo(f"\nDone. Play all: for f in {out}/*.wav; do echo $f; afplay $f; done")


if __name__ == "__main__":
    cli()
