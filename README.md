# qwen3-tts-spanish-voices

Curated Spanish TTS voices powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) ([HuggingFace models](https://huggingface.co/collections/Qwen/qwen3-tts)) running locally on Apple Silicon via MLX.

14 voices out of the box — 12 cloned from real speakers (VoxForge Spanish corpus) covering Spain, Mexico, Argentina, Chile, and Ibero-American accents, plus 2 designed voices for quick use without reference audio.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4) with ≥16GB RAM (32GB+ recommended)
- Python 3.11+
- Conda environment with MLX Audio (`mlx-audio>=0.3.0`)

## Installation

```bash
# Create/activate the environment
conda activate qwen3-tts

# Install in editable mode
cd ~/Code/qwen3-tts-spanish-voices  # or wherever you cloned it
pip install -e ".[mlx]"
```

## Quick Start

```bash
# Generate with default voice (neutral_male, design mode)
spanish-tts say "Hola mundo, esto es una prueba de texto a voz."

# Use a specific voice
spanish-tts say "Buenos días" --voice carlos_mx

# Generate and auto-play
spanish-tts say "El café colombiano es magnífico." --voice elena_mx --play

# Adjust speed (0.8 = slow, 1.3 = fast)
spanish-tts say "Rápido como el viento" --voice energetic_male --speed 1.2

# Custom output path
spanish-tts say "Guardado aquí" --voice warm_female --output ~/my-audio.wav
```

## Available Voices

### Clone Voices (from VoxForge Spanish)

| Name           | Gender | Accent        | Source   |
|----------------|--------|---------------|----------|
| pedro_es       | Male   | Spain         | VoxForge |
| lucia_es       | Female | Spain         | VoxForge |
| carlos_mx      | Male   | Mexico        | VoxForge |
| elena_mx       | Female | Mexico        | VoxForge |
| martin_ar      | Male   | Argentina     | VoxForge |
| sofia_ar       | Female | Argentina     | VoxForge |
| mateo_cl       | Male   | Chile         | VoxForge |
| camila_cl      | Female | Chile         | VoxForge |
| diego_la       | Male   | Ibero-America | VoxForge |
| valentina_la   | Female | Ibero-America | VoxForge |
| neutral_female | Female | Ibero-America | VoxForge |
| warm_female    | Female | Spain         | VoxForge |

### Design Voices (no reference audio needed)

| Name           | Gender | Description                          |
|----------------|--------|--------------------------------------|
| neutral_male   | Male   | Clear, calm narrator                 |
| energetic_male | Male   | Upbeat, dynamic podcast host         |

> **Note:** Design voices use English instruct prompts with explicit `lang_code="spanish"` to ensure native prosody. Clone voices produce more natural accents and are recommended when quality matters most.

## CLI Commands

```bash
# List all voices
spanish-tts list

# Generate speech
spanish-tts say "text" --voice NAME [--speed N] [--output PATH] [--play]

# Demo: generate same text with ALL voices for comparison
spanish-tts demo "El café colombiano es reconocido mundialmente."

# Add a custom clone voice from your own audio
spanish-tts add-ref my_voice /path/to/audio.wav "transcript of the audio" --accent colombia --gender male

# Add a designed voice from a description
spanish-tts add-design narrator "A 50-year-old male with deep baritone voice, very slow pace." --gender male

# Remove a voice
spanish-tts remove my_voice
```

## Adding Your Own Voices

### Clone from audio (best quality)

Provide a clean 5-10 second recording and its transcript:

```bash
spanish-tts add-ref abuela ~/recordings/abuela.wav \
  "Y entonces tu abuelo me dijo que fuéramos al parque" \
  --accent colombia --gender female
```

### Design from description

No audio needed — describe the voice you want:

```bash
spanish-tts add-design profesor \
  "A 55-year-old Colombian man with a deep, authoritative voice. Speaks slowly and clearly, like a university professor giving a lecture." \
  --gender male
```

## Architecture

```text
qwen3-tts-spanish-voices/
├── src/spanish_tts/
│   ├── cli.py       # Click CLI (say, list, demo, add-ref, add-design, remove)
│   ├── config.py    # YAML voice registry management
│   └── engine.py    # MLX Qwen3-TTS wrapper (clone + design generation)
├── presets/
│   └── voices.yaml  # Default voice definitions (shipped with package)
├── scripts/
│   └── curate.py    # VoxForge corpus browser for finding reference audio
└── pyproject.toml
```

## Configuration

Voice registry lives at `~/.spanish-tts/voices.yaml`. Reference audio files are stored in `~/.spanish-tts/references/`.

Generated audio goes to `~/tts-output/spanish/` by default (configurable in voices.yaml under `defaults.output_dir`).

## Models Used

| Mode   | Model                                              | Size  |
|--------|----------------------------------------------------|-------|
| Clone  | mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit        | 2.9GB |
| Design | mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | 2.9GB |

Models are downloaded automatically on first use to `~/.cache/huggingface/hub/`.

## Performance (M1 Max 64GB)

- First run: ~5s model load + generation
- Subsequent runs (model cached): ~6s for a typical sentence
- Design voices: slightly faster (no audio encoding step)
- Clone voices: slightly slower (encodes reference first)

## Data Source

Clone voices sourced from [VoxForge Spanish (CIEMPIESS)](https://huggingface.co/datasets/ciempiess/voxforge_spanish) — a Creative Commons licensed corpus of read Spanish speech covering multiple regional accents.

## License

MIT
