"""Audio chain of buzz-video-content: the voice cut like the picture, original BGM, SFX sets, the mix and its mastering.

  voice.py   source audio -> dialogue / ambience tracks on the output timeline (Rubber Band, EQ, denoise only if noisy)
  music.py   BGM for 15 s - 20 min: form.py (plan) + style_<lofi|acoustic|pop|wa|ambient|piano>.py (+ theory, melody,
             arrange, instruments, wa_*)
  sfx.py     named one-shots in five sets (soft, cute, variety, wa, none)
  mix.py     levels by measurement (voice -16 LUFS, music under speech, swells), -14 LUFS master, true-peak limit
  dsp.py     filters, reverb, BS.1770 loudness, 4x-oversampled true peak and the limiter built on it
scripts/finish.py drives the chain for one version; scripts/qa.py measures the result.
"""
