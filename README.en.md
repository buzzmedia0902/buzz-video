# buzz-video

[日本語](README.md)

**Claude Code skills that edit social videos for you.** Give Claude your phone or camera footage and it edits and exports videos for posting and promotion. The skills, their questions and the on-screen text are in Japanese.

| Skill | What it makes |
|---|---|
| **buzz-video-content** | YouTube videos (16:9), Shorts / Reels / TikTok (9:16), clips cut from long videos, everyday vlogs |
| **buzz-video-ad** | Social ads and Reels for shops, salons, clinics and restaurants (9:16 / 4:5 / 1:1 / 16:9) |

## Features
- **Asks first, with multiple-choice questions**: use, tone, aspect ratio and length. Give it a reference video and it measures cut rate, speaking pace and caption placement, then matches them.
- **Five tones for talk videos**: trust (explainer), friendly, variety, wa (Japanese brush style) and minimal. A tone switches captions, pause trimming, camera moves, music and sound effects together.
- **Five tastes for ads**: standard, motion, cinematic, narrated vlog and a chat-screen story (no LINE logo or name).
- Transcription (Whisper), jump cuts that remove pauses, captions, explainer cards, chapter titles and an end screen. Claude writes an edit list and the engine renders it.
- Captions and cards avoid faces (Apple Vision). Speakers are identified by voice and colour-coded; a line it cannot identify gets no name.
- **Music and sound effects are composed and synthesized on the spot** (no existing songs or sample packs).
- Loudness -14 LUFS with true peak at or below -1 dBTP, a light copy for phones, SRT subtitles, three thumbnail options, a YouTube description or post text.
- Ads check industry rules (medical, acupuncture, cosmetics, ...) while the copy is written.

## Requirements
- **macOS** (Apple Silicon recommended): Apple Vision for faces, VideoToolbox for decoding. Needs [Homebrew](https://brew.sh) (for ffmpeg, whisper-cpp, uv, ...).
- **Windows 10 / 11 (64-bit)**: needs [Git for Windows](https://gitforwindows.org/) (Claude Code runs the skills' steps in Git Bash). Faces come from MediaPipe; the setup unpacks the tools under `~/.local/opt`. Rendering is slower than on a Mac.
- [Claude Code](https://claude.com/claude-code)

## Install

### Option 1: as a Claude Code plugin (recommended)
```
/plugin marketplace add buzzmedia0902/buzz-video
/plugin install buzz-video@buzz-video
```

### Option 2: into your skills folder
```bash
git clone https://github.com/buzzmedia0902/buzz-video.git ~/buzz-video
mkdir -p ~/.claude/skills
ln -sfn ~/buzz-video/skills/buzz-video-content ~/.claude/skills/buzz-video-content
ln -sfn ~/buzz-video/skills/buzz-video-ad ~/.claude/skills/buzz-video-ad
```
On Windows, in PowerShell (no admin rights needed):
```powershell
git clone https://github.com/buzzmedia0902/buzz-video.git $HOME\buzz-video
New-Item -ItemType Directory -Force $HOME\.claude\skills | Out-Null
New-Item -ItemType Junction -Path $HOME\.claude\skills\buzz-video-content -Target $HOME\buzz-video\skills\buzz-video-content
New-Item -ItemType Junction -Path $HOME\.claude\skills\buzz-video-ad -Target $HOME\buzz-video\skills\buzz-video-ad
```

### First-time setup
Ask Claude to "set up buzz-video" (or run `bash ~/buzz-video/setup.sh`). It installs only what is missing:
- tools: ffmpeg, whisper-cpp, uv (required); rubberband, yt-dlp (optional), from Homebrew on a Mac and from their publishers into `~/.local/opt` on Windows
- the Whisper model (about 550 MB)
- a free font: M PLUS Rounded 1c
- a private Python environment under `~/.local/share/` (on Windows also one for face detection)

Some fonts that come with macOS (Tsukushi Round Gothic, Toppan Bunkyu Midashi, Kaiti, Xingkai, ...) must be downloaded once in Font Book. Without them the skills still work, with plainer type. Windows uses Yu Gothic, Yu Mincho and Meiryo instead. `bash setup.sh --check` only reports and installs nothing.

## Use
Tell Claude where the footage is, for example "edit ~/Movies/talk.mov for YouTube", "make three Shorts from this video" or "make a Reels ad from these clips and our logo". Claude looks at the footage, asks its questions, checks stills, and shares a light copy of the first video before finishing the rest.

As a plugin, the skills can also be called as `/buzz-video:buzz-video-content` and `/buzz-video:buzz-video-ad`.

## Narration (optional, ad vlog taste)
- Uses a local Japanese TTS engine: put the macOS build of [AivisSpeech Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) or [VOICEVOX Engine](https://github.com/VOICEVOX/voicevox_engine) in `~/.local/opt/aivisspeech_engine-<version>/` or `~/.local/opt/voicevox_engine-<version>/` (the folder that holds `run`).
- The default voice is AivisSpeech "morioki"; if it is not installed, Claude picks another installed voice (voice models can be added in AivisSpeech's model manager or from AivisHub).
- Check each voice model's terms (commercial use, credit, prohibited uses). VOICEVOX requires the credit "VOICEVOX:<character>".

## Please note
- Clip or download only videos you have the rights to use (your own appearance, a licensed show, ...), and follow the original channel's clipping guidelines.
- Before posting, check the consent of the people on screen and the rights to any music or logos in the footage.
- The notes on advertising rules are general guidance, not legal advice.
- Rendering, transcription and music run on your Mac. The stills and transcripts Claude looks at while checking are sent to Anthropic as part of the conversation. Otherwise the network is used only to fetch tools and models, to download a video from a URL, or for a cloud voice you choose.
- LINE is a third-party trademark. This project is not affiliated with LINE.

## How it works
Claude writes an edit list (`edit.json` / `project.json`); a Python engine (skia, Pillow, numpy) and ffmpeg draw every frame, and numpy composes the music. Each skill's `SKILL.md` is the workflow, `references/` holds the detailed rules and `scripts/` is the engine. See [scripts/DEV.md](skills/buzz-video-content/scripts/DEV.md) for buzz-video-content's internals.

## License
MIT License ([LICENSE](LICENSE)) © 2026 BUZZ MEDIA
