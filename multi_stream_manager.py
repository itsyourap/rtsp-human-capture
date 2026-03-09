import os
import threading
from typing import Dict, List, Optional, Union

import cv2

from display_manager import DisplayManager
from person_detector import PersonDetector
from stream_processor import StreamProcessor


class MultiStreamManager:
  """Orchestrates concurrent processing of multiple RTSP streams.

  Each stream runs in its own daemon thread via :class:`StreamProcessor`.
  When *display* is ``True``, a single :class:`DisplayManager` grid window
  shows all streams simultaneously.

  Example::

      detector = PersonDetector()
      manager = MultiStreamManager(detector)
      manager.process_multiple_streams(
          rtsp_urls=["rtsp://cam1/stream", "rtsp://cam2/stream"],
          frame_skip=15,
          save_mode="video",
          display=True,
      )
  """

  def __init__(self, detector: PersonDetector, output_dir: str = "person") -> None:
    self.detector = detector
    self.output_dir = output_dir
    self._processor = StreamProcessor(detector, output_dir=output_dir)

  def process_multiple_streams(
    self,
    rtsp_urls: Union[List[str], Dict[Union[int, str], str]],
    frame_skip: int = 15,
    save_mode: Optional[str] = None,
    display: bool = False,
  ) -> None:
    """Start all streams and block until they finish (or Ctrl-C).

    Args:
        rtsp_urls: List of RTSP URLs (auto-numbered 1…N) or a ``{id: url}`` dict.
        frame_skip: Analyse every Nth frame.
        save_mode: ``'image'`` / ``'video'`` / ``None``.
        display: Show a live grid window with all streams.
    """
    if save_mode is not None:
      os.makedirs(self.output_dir, exist_ok=True)
      print(f"Created main directory: {self.output_dir}")

    # Build ordered (stream_id, url) list
    if isinstance(rtsp_urls, dict):
      stream_list = list(rtsp_urls.items())
    else:
      stream_list = list(enumerate(rtsp_urls, 1))

    num_streams = len(stream_list)
    print(f"Starting person detection on {num_streams} stream(s)...")

    # Optionally start the grid display
    display_manager: Optional[DisplayManager] = None
    if display:
      display_manager = DisplayManager()
      display_manager.start([sid for sid, _ in stream_list])

    # Launch one worker thread per stream
    threads: List[threading.Thread] = []
    for stream_id, rtsp_url in stream_list:
      t = threading.Thread(
        target=self._processor.process_single_stream,
        args=(stream_id, rtsp_url, frame_skip, save_mode, display_manager),
        daemon=True,
      )
      t.start()
      threads.append(t)
      print(f"Started thread for stream {stream_id}: {rtsp_url}")

    try:
      if display:
        print("All streams shown in a single grid window. Press 'q' or Ctrl+C to stop...")
      else:
        print("All streams started. Press Ctrl+C to stop all streams...")

      for t in threads:
        while t.is_alive():
          t.join(timeout=0.5)
          if display and display_manager is not None and not display_manager.is_running:
            break
        if display and display_manager is not None and not display_manager.is_running:
          break

    except KeyboardInterrupt:
      print("\nStopping all streams...")

    finally:
      if display_manager is not None:
        display_manager.stop()
      else:
        cv2.destroyAllWindows()
