"""Application configuration loaded from a .cfg (INI-style) file.

Typical usage::

    from config import load_config

    cfg = load_config()                  # reads config.cfg in cwd
    cfg = load_config("path/to/my.cfg")  # custom path
"""

import configparser
import os
from dataclasses import dataclass

DEFAULT_CONFIG_PATH = "config.cfg"


@dataclass
class AppConfig:
  # [paths]
  model_dir: str
  output_dir: str

  # [detection]
  confidence_threshold: float
  person_area_threshold: int
  frame_skip: int


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
  """Load *path* and return a validated :class:`AppConfig`.

  Missing keys fall back to the same defaults that are written in the
  bundled ``config.cfg`` so the application works even with a minimal file.

  Raises:
      FileNotFoundError: if *path* does not exist.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(
      f"Config file not found: '{path}'. "
      f"Create one or copy the default config.cfg."
    )

  parser = configparser.ConfigParser()
  parser.read(path)

  model_dir = parser.get("paths", "model_dir", fallback="model").strip()
  output_dir = parser.get("paths", "output_dir", fallback="person").strip()

  confidence_threshold = parser.getfloat(
    "detection", "confidence_threshold", fallback=0.5
  )
  person_area_threshold = parser.getint(
    "detection", "person_area_threshold", fallback=1000
  )
  frame_skip = parser.getint("detection", "frame_skip", fallback=15)

  # Basic validation
  if not 0.0 < confidence_threshold < 1.0:
    raise ValueError(
      f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
    )
  if person_area_threshold < 0:
    raise ValueError(
      f"person_area_threshold must be >= 0, got {person_area_threshold}"
    )
  if frame_skip < 1:
    raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

  return AppConfig(
    model_dir=model_dir,
    output_dir=output_dir,
    confidence_threshold=confidence_threshold,
    person_area_threshold=person_area_threshold,
    frame_skip=frame_skip,
  )
