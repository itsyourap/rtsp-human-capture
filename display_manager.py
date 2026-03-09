import math
import threading
from typing import Dict, List, Optional, Union

import cv2
import numpy as np


class DisplayManager:
  """Manages a single resizable grid window that composites frames from
  multiple streams in real time.

  Usage::

      dm = DisplayManager()
      dm.start([1, 2, 3])          # register stream ids and open window
      dm.update_frame(1, frame)    # called from each stream thread
      ...
      dm.stop()
  """

  def __init__(self, cell_width: int = 640, cell_height: int = 360) -> None:
    self.cell_width = cell_width
    self.cell_height = cell_height

    self._frames: Dict[Union[int, str], Optional[cv2.typing.MatLike]] = {}
    self._lock: threading.Lock = threading.Lock()
    self._running: bool = False
    self._stream_ids: List[Union[int, str]] = []
    self._thread: Optional[threading.Thread] = None

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  @property
  def is_running(self) -> bool:
    return self._running

  def start(self, stream_ids: List[Union[int, str]]) -> None:
    """Register *stream_ids* and start the display thread."""
    self._stream_ids = list(stream_ids)
    with self._lock:
      for sid in stream_ids:
        self._frames[sid] = None  # pyright: ignore[reportArgumentType]
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    """Signal the display thread to stop and wait for it to finish."""
    self._running = False
    if self._thread is not None:
      self._thread.join(timeout=2)
    cv2.destroyAllWindows()
    with self._lock:
      self._frames.clear()

  def update_frame(self, stream_id: Union[int, str], frame: cv2.typing.MatLike) -> None:
    """Update the display buffer for *stream_id*. Thread-safe."""
    with self._lock:
      self._frames[stream_id] = frame

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _build_grid(self) -> cv2.typing.MatLike:
    """Compose all buffered frames into a single grid image."""
    with self._lock:
      ids = list(self._stream_ids)
      frames = {sid: self._frames.get(sid) for sid in ids}

    n = len(ids)
    if n == 0:
      return np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid_rows = []
    idx = 0
    for _ in range(rows):
      row_cells = []
      for _ in range(cols):
        if idx < n:
          f = frames[ids[idx]]
          if f is not None:
            cell = cv2.resize(f, (self.cell_width, self.cell_height))
          else:
            cell = np.zeros(
              (self.cell_height, self.cell_width, 3), dtype=np.uint8)
            cv2.putText(
              cell,
              f"Stream {ids[idx]} - Connecting...",
              (10, self.cell_height // 2),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.7,
              (100, 100, 100),
              2,
            )
        else:
          cell = np.zeros(
            (self.cell_height, self.cell_width, 3), dtype=np.uint8)
        row_cells.append(cell)
        idx += 1
      grid_rows.append(np.hstack(row_cells))

    return np.vstack(grid_rows)

  def _loop(self) -> None:
    """Display thread: renders the grid at ~30 fps until stopped."""
    window_name = "Person Detection - All Streams"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while self._running:
      grid = self._build_grid()
      cv2.imshow(window_name, grid)
      key = cv2.waitKey(30) & 0xFF
      if key == ord("q"):
        self._running = False
        break

    cv2.destroyWindow(window_name)
