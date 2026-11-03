"""Thumbnails and covers: speaker cut-out, tone themes, three YouTube directions, 9:16 covers, feed preview.

  cutout   Apple Vision matte -> clean alpha -> sticker (white outline + thin black edge + drop shadow)
  theme    colours / fonts / decoration of a tone, read from its Look class
  draw     type, badges, speech bubbles, cards, marks, backgrounds (skia)
  layouts  reaction / before_after / authority (1280x720) and the cover (1080x1920)
  scan     frames of a talk every ~1.5 s, sharpest frame near a time, face crops, role picks
  feed     phone-feed mock with the real title
"""
