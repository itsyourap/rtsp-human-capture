import os
import time
from typing import List, Optional, Tuple, Union

import cv2

from display_manager import DisplayManager
from person_detector import PersonDetector


class StreamProcessor:
  """Processes a single RTSP stream, runs person detection, handles saving
  (image snapshots or video clips) and feeds frames to an optional
  :class:`DisplayManager`.

  Intended to be instantiated once per stream and started in its own thread::

      processor = StreamProcessor(detector)
      thread = threading.Thread(
          target=processor.process_single_stream,
          args=(stream_id, rtsp_url, frame_skip, save_mode, display_manager),
      )
      thread.start()
  """

  def __init__(self, detector: PersonDetector, output_dir: str = "person") -> None:
    self.detector = detector
    self.output_dir = output_dir

  # ------------------------------------------------------------------
  # Multi-stream entry point (worker function, called from a thread)
  # ------------------------------------------------------------------

  def process_single_stream(
    self,
    stream_id: Union[int, str],
    rtsp_url: str,
    frame_skip: int = 15,
    save_mode: Optional[str] = None,
    display_manager: Optional[DisplayManager] = None,
  ) -> None:
    """Process *rtsp_url* indefinitely in the calling thread.

    Args:
        stream_id: Unique identifier for this stream.
        rtsp_url: RTSP stream URL.
        frame_skip: Analyse every Nth frame.
        save_mode: ``'image'`` for JPEG snapshots, ``'video'`` for MP4 clips,
                   ``None`` to disable saving.
        display_manager: If provided, updates the shared grid display buffer
                         with annotated frames.
    """
    print(f"[Stream {stream_id}] Connecting to: {rtsp_url}")

    if save_mode is not None:
      person_dir = f"{self.output_dir}/stream_{stream_id}"
      os.makedirs(person_dir, exist_ok=True)
      print(f"[Stream {stream_id}] Created directory: {person_dir}")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
      print(f"[Stream {stream_id}] Error: Could not connect to RTSP stream")
      return

    print(f"[Stream {stream_id}] Connected successfully! Processing frames...")

    frame_count: int = 0
    person_present: bool = False
    person_entry_count: int = 0
    last_detection_time: float = 0
    consecutive_failures: int = 0
    max_reconnect_attempts: int = 5

    # Clip recording state
    video_writer: Optional[cv2.VideoWriter] = None
    no_person_streak: int = 0
    NO_PERSON_EXIT_THRESHOLD: int = 3
    clip_filename: str = ""
    stream_fps: float = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0 or stream_fps > 120:
      stream_fps = 25.0

    # Last known detection results (re-used for display between analysis frames)
    current_person_count: int = 0
    current_boxes: List[Tuple[int, int, int, int, float]] = []

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
          consecutive_failures += 1
          print(
            f"[Stream {stream_id}] Failed to read frame "
            f"(attempt {consecutive_failures}/{max_reconnect_attempts}), reconnecting..."
          )
          cap.release()
          time.sleep(2)
          cap = cv2.VideoCapture(rtsp_url)
          if not cap.isOpened():
            if consecutive_failures >= max_reconnect_attempts:
              print(
                f"[Stream {stream_id}] Max reconnect attempts reached. Giving up.")
              break
            continue
          print(f"[Stream {stream_id}] Reconnected successfully.")
          continue

        consecutive_failures = 0
        frame_count += 1
        current_time = time.time()

        # ---------- Detection (every Nth frame) ----------
        if frame_count % frame_skip == 0 and (current_time - last_detection_time) >= 0.5:
          last_detection_time = current_time
          has_person, person_count, boxes = self.detector.detect_persons(frame)
          current_person_count = person_count
          current_boxes = boxes

          timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
          status = f"{person_count} person(s) detected" if has_person else "No persons"
          print(f"[{timestamp}] [Stream {stream_id}] Frame {frame_count}: {status}")

          if has_person and not person_present:
            # Person entered the frame
            person_present = True
            no_person_streak = 0
            person_entry_count += 1
            print(
              f"[Stream {stream_id}]   Person entered frame! Entry #{person_entry_count}")
            print(
              f"[Stream {stream_id}]   Detected {person_count} person(s) with boxes: "
              f"{[(x, y, w, h) for x, y, w, h, _ in boxes]}"
            )

            if save_mode is not None:
              person_dir = f"{self.output_dir}/stream_{stream_id}"
              os.makedirs(person_dir, exist_ok=True)
              timestamp_str = time.strftime("%Y%m%d_%H%M%S")

              if save_mode == "image":
                filename = (
                  f"{person_dir}/person_entry_{person_entry_count}"
                  f"_{timestamp_str}_{int(time.time())}.jpg"
                )
                self._save_annotated_snapshot(frame, boxes, filename)
                print(f"[Stream {stream_id}]   Saved snapshot: {filename}")

              elif save_mode == "video":
                clip_filename = (
                  f"{person_dir}/person_clip_{person_entry_count}"
                  f"_{timestamp_str}_{int(time.time())}.mp4"
                )
                h_frame, w_frame = frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                  clip_filename, fourcc, stream_fps, (w_frame, h_frame)
                )
                print(
                  f"[Stream {stream_id}]   Started recording clip: {clip_filename}")

          elif has_person and person_present:
            no_person_streak = 0
            if frame_count % (frame_skip * 20) == 0:
              print(f"[Stream {stream_id}]   Person(s) still present in frame")

          elif not has_person and person_present:
            no_person_streak += 1
            print(
              f"[Stream {stream_id}]   No person detected "
              f"({no_person_streak}/{NO_PERSON_EXIT_THRESHOLD})"
            )
            if no_person_streak >= NO_PERSON_EXIT_THRESHOLD:
              person_present = False
              no_person_streak = 0
              if video_writer is not None:
                video_writer.release()
                video_writer = None
                print(
                  f"[Stream {stream_id}]   Person(s) exited. Saved clip: {clip_filename}")
              print(
                f"[Stream {stream_id}]   Person(s) exited frame. Waiting for next entry...")

        # ---------- Write clip frame ----------
        if video_writer is not None:
          video_writer.write(frame)

        # ---------- Update grid display ----------
        if display_manager is not None:
          display_frame = frame.copy()
          for x, y, w, h, confidence in current_boxes:
            cv2.rectangle(display_frame, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
              display_frame,
              f"Person {confidence:.2f}",
              (x, y - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (0, 255, 0),
              2,
            )
          status_text = (
            f"Stream {stream_id} | Persons: {current_person_count} | Entries: {person_entry_count}"
          )
          cv2.putText(display_frame, status_text, (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          display_manager.update_frame(stream_id, display_frame)

          if not display_manager.is_running:
            break

    except KeyboardInterrupt:
      print(f"[Stream {stream_id}] Stopping detection...")

    finally:
      if video_writer is not None:
        video_writer.release()
        print(f"[Stream {stream_id}] Saved in-progress clip: {clip_filename}")
      cap.release()
      saved_noun = "snapshot(s)" if save_mode == "image" else "clip(s)"
      print(
        f"[Stream {stream_id}] Processed {frame_count} frames, "
        f"captured {person_entry_count} person {saved_noun}"
      )

  # ------------------------------------------------------------------
  # Single-stream entry point (its own display window)
  # ------------------------------------------------------------------

  def process_rtsp_stream(
    self,
    rtsp_url: str,
    frame_skip: int = 15,
    display: bool = True,
    save_mode: Optional[str] = None,
  ) -> None:
    """Process a single RTSP stream with an optional dedicated display window.

    Args:
        rtsp_url: RTSP stream URL.
        frame_skip: Analyse every Nth frame (default 15 ≈ 2 fps on a 30 fps stream).
        display: Show a live annotated window (resized to 1280×720).
        save_mode: ``'image'`` / ``'video'`` / ``None``.
    """
    print(f"Connecting to RTSP stream: {rtsp_url}")

    if save_mode is not None:
      os.makedirs(self.output_dir, exist_ok=True)
      print(f"Created directory: {self.output_dir}")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
      print("Error: Could not connect to RTSP stream")
      return

    print("Connected successfully! Processing frames...")
    if display:
      print("Press 'q' to quit")

    frame_count: int = 0
    person_present: bool = False
    person_entry_count: int = 0
    last_detection_time: float = 0
    consecutive_failures: int = 0
    max_reconnect_attempts: int = 5

    video_writer: Optional[cv2.VideoWriter] = None
    no_person_streak: int = 0
    NO_PERSON_EXIT_THRESHOLD: int = 3
    clip_filename: str = ""
    stream_fps: float = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0 or stream_fps > 120:
      stream_fps = 25.0

    has_person: bool = False
    person_count: int = 0
    boxes: List[Tuple[int, int, int, int, float]] = []

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
          consecutive_failures += 1
          print(
            f"Failed to read frame (attempt {consecutive_failures}/{max_reconnect_attempts}), reconnecting..."
          )
          cap.release()
          time.sleep(2)
          cap = cv2.VideoCapture(rtsp_url)
          if not cap.isOpened():
            if consecutive_failures >= max_reconnect_attempts:
              print("Max reconnect attempts reached. Giving up.")
              break
            continue
          print("Reconnected successfully.")
          continue

        consecutive_failures = 0
        frame_count += 1
        current_time = time.time()

        # ---------- Detection ----------
        if frame_count % frame_skip == 0 and (current_time - last_detection_time) >= 0.5:
          last_detection_time = current_time
          has_person, person_count, boxes = self.detector.detect_persons(frame)

          timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
          status = f"{person_count} person(s) detected" if has_person else "No persons"
          print(f"[{timestamp}] Frame {frame_count}: {status}")

          if has_person and not person_present:
            person_present = True
            no_person_streak = 0
            person_entry_count += 1
            print(f"  Person entered frame! Entry #{person_entry_count}")

            if save_mode is not None:
              os.makedirs(self.output_dir, exist_ok=True)
              timestamp_str = time.strftime("%Y%m%d_%H%M%S")

              if save_mode == "image":
                filename = (
                  f"{self.output_dir}/person_entry_{person_entry_count}"
                  f"_{timestamp_str}_{int(time.time())}.jpg"
                )
                self._save_annotated_snapshot(frame, boxes, filename)
                print(f"  Saved snapshot: {filename}")

              elif save_mode == "video":
                clip_filename = (
                  f"{self.output_dir}/person_clip_{person_entry_count}"
                  f"_{timestamp_str}_{int(time.time())}.mp4"
                )
                h_frame, w_frame = frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                  clip_filename, fourcc, stream_fps, (w_frame, h_frame)
                )
                print(f"  Started recording clip: {clip_filename}")

          elif has_person and person_present:
            no_person_streak = 0

          elif not has_person and person_present:
            no_person_streak += 1
            print(
              f"  No person detected ({no_person_streak}/{NO_PERSON_EXIT_THRESHOLD})")
            if no_person_streak >= NO_PERSON_EXIT_THRESHOLD:
              person_present = False
              no_person_streak = 0
              if video_writer is not None:
                video_writer.release()
                video_writer = None
                print(f"  Person(s) exited. Saved clip: {clip_filename}")
              print("  Person(s) exited frame. Waiting for next entry...")

        # ---------- Write clip frame ----------
        if video_writer is not None:
          video_writer.write(frame)

        # ---------- Display ----------
        if display:
          display_frame = cv2.resize(frame.copy(), (1280, 720))

          if frame_count % frame_skip == 0:
            original_height, original_width = frame.shape[:2]
            scale_x = 1280 / original_width
            scale_y = 720 / original_height

            for x, y, w, h, confidence in boxes:
              sx, sy = int(x * scale_x), int(y * scale_y)
              sw, sh = int(w * scale_x), int(h * scale_y)
              cv2.rectangle(display_frame, (sx, sy),
                            (sx + sw, sy + sh), (0, 255, 0), 2)
              cv2.putText(
                display_frame,
                f"Person {confidence:.2f}",
                (sx, sy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
              )

          cv2.putText(
            display_frame,
            f"Persons: {person_count} | Entries: {person_entry_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
          )
          cv2.imshow("RTSP Person Detection", display_frame)
          if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except KeyboardInterrupt:
      print("\nStopping detection...")

    finally:
      if video_writer is not None:
        video_writer.release()
        print(f"Saved in-progress clip: {clip_filename}")
      cap.release()
      if display:
        cv2.destroyAllWindows()
      saved_noun = "snapshot(s)" if save_mode == "image" else "clip(s)"
      print(
        f"Processed {frame_count} frames, captured {person_entry_count} person {saved_noun}")

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _save_annotated_snapshot(
    frame: cv2.typing.MatLike,
    boxes: List[Tuple[int, int, int, int, float]],
    filename: str,
  ) -> None:
    annotated = frame.copy()
    for x, y, w, h, confidence in boxes:
      cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(
        annotated,
        f"Person {confidence:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
      )
    cv2.imwrite(filename, annotated)
