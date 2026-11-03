# buzz-video-content engine (for maintainers and helper agents)

Pipeline: `edit.json` -> `render.py` (silent master, stills, sheets; writes `work/<version>/cues_out.json`)
-> `finish.py` (voice cut like the picture, BGM, SFX, mix -14 LUFS, mux, light copy, SRT, 概要欄, report).

| file | role |
|---|---|
| `cv/canvas.py` | W, H, FPS of the version (from env CV_ASPECT / CV_FPS, set by render.py before imports), safe area |
| `cv/spec.py` | edit.json load / validation (sources, versions, parts, overlays) |
| `cv/vad.py` | speech mask (band energy), bursts, jet cut (`tighten`) |
| `cv/timeline.py` | parts -> placed `Shot`s (talk = jet cut, clip = range, card = generated); src <-> output time |
| `cv/framing.py` | crop / box per shot (cover or band), wide <-> medium camera plan, face track loader |
| `cv/reader.py` | ffmpeg decode of a shot (speed, fps, rotation, HLG LUT, crop, scale, grade) -> RGBA |
| `cv/cues.py` | cues.json -> output-time subtitles (tidy fillers, min show 1.2 s, joins), SRT |
| `cv/overlays.py` | authored overlays + format defaults (title band, credit) in output time |
| `cv/layout.py` | face-aware placement (cards / name plates / big words), 9:16 card mode |
| `cv/gfx.py` | skia kit: easing, paints, fonts by role key (`FAMILIES`), outlined text, images |
| `cv/linebreak.py` | Japanese line-break costs (janome) |
| `cv/tones/base.py` | `Look`: pacing (CUT / CAMERA / AUDIO / GRADE / SUBS) + every draw_* element; tones subclass it |
| `cv/tones/<tone>.py` | one tone each, exports `LOOK` |
| `render.py` | `Comp` (one frame = footage or card + overlays), stills / sheet / parallel video |
| `transcribe.py` | per-burst Whisper -> draft cues.json; `--align` re-times corrected cues |
| `faces.py` | Apple Vision face boxes at 2 fps -> work/faces/<src>.jsonl |
| `speakers.py` + `cv/speakers_lib.py` | voice ID (ECAPA, separate torch env via `env.sh --speakers`): tags cues only when sure |
| `find.py` | ranks 切り抜き candidates in a long transcribed video; `--download URL` (yt-dlp fallbacks) |
| `probe.py` | footage inventory, contact sheets, speech share, optional flash scan |
| `thumb.py` + `cv/thumbs/` | YouTube thumbnails (3 directions) / 9:16 covers in the tone's look; face scan, Vision cut-out, feed preview |
| `ref_probe.py` + `cv/refprobe/` | measures a reference video (cuts, pauses, speech rate, loudness, telops) -> nearest tone + `look` |
| `finish.py` | voice cut like the picture + BGM + SFX + mix (-14 LUFS, TP <= -1 dBTP on the AAC), mux, light copy, SRT, 概要欄, cover, report |
| `qa.py` | loudness, frames, judder, subtitle timing / reading speed, shot changes and punch-ins per minute |
| `cv/audio/` | dsp (LUFS, true peak, limiter), music (original BGM: styles lofi / acoustic / pop / piano / ambient / wa, any length, exact bpm), sfx, voice, mix |
| `tools/*.swift` | Apple Vision helpers compiled on first use into ~/.local/share/content-video/bin (faces, landmarks, lift, textboxes) |
| tone `THUMB = {...}` | optional per-tone poster palette for thumbnails (cv/thumbs/theme.py) |

Conventions: every module draws in canvas px from fractions (works for 9:16 / 4:5 / 1:1 / 16:9); tones change
class attributes first and override draw methods only where the look differs; times in overlays are output
seconds; `{..}` marks emphasis and `｜` a forced line break in any text.
