import argparse
import os
import time

import cv2

from multi_stream_manager import MultiStreamManager
from person_detector import PersonDetector
from stream_processor import StreamProcessor


def test_with_image(detector: PersonDetector, image_path: str) -> None:
  """Test the detector with a single image file and save an annotated result."""
  print(f"Testing with image: {image_path}")

  if not os.path.exists(image_path):
    print(f"Error: Image file {image_path} not found")
    return

  frame = cv2.imread(image_path)
  if frame is None:
    print(f"Error: Could not load image {image_path}")
    return

  has_person, person_count, boxes = detector.detect_persons(frame)
  print(f"Persons detected: {person_count}")
  print(f"Bounding boxes: {boxes}")

  for i, (x, y, w, h, confidence) in enumerate(boxes):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
      frame,
      f"Person {i + 1}: {confidence:.2f}",
      (x, y - 10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 255, 0),
      2,
    )

  output_path = f"test_result_{int(time.time())}.jpg"
  cv2.imwrite(output_path, frame)
  print(f"Annotated result saved to: {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Person Detection for RTSP Streams")
  parser.add_argument("--rtsp", type=str, help="Single RTSP stream URL")
  parser.add_argument("--rtsp-list", nargs="+",
                      help="Multiple RTSP stream URLs")
  parser.add_argument("--rtsp-file", type=str,
                      help="File containing RTSP URLs (one per line)")
  parser.add_argument("--test-image", type=str,
                      help="Test with image file instead of RTSP")
  parser.add_argument("--confidence", type=float, default=0.5,
                      help="Confidence threshold (default: 0.5)")
  parser.add_argument("--area-threshold", type=int, default=1000,
                      help="Minimum person area in pixels (default: 1000)")
  parser.add_argument("--frame-skip", type=int, default=15,
                      help="Process every Nth frame (default: 15)")
  parser.add_argument("--no-display", action="store_true",
                      help="Disable video display for single stream mode")
  parser.add_argument("--display", action="store_true",
                      help="Enable grid display for multiple streams")
  parser.add_argument(
    "--save",
    type=str,
    choices=["image", "video"],
    required=True,
    help="Save mode: 'image' for snapshot JPGs, 'video' for MP4 clips",
  )

  args = parser.parse_args()

  detector = PersonDetector(
    confidence_threshold=args.confidence,
    person_area_threshold=args.area_threshold,
  )

  if args.test_image:
    test_with_image(detector, args.test_image)

  elif args.rtsp_list:
    print(f"Processing {len(args.rtsp_list)} RTSP streams...")
    manager = MultiStreamManager(detector)
    manager.process_multiple_streams(
      rtsp_urls=args.rtsp_list,
      frame_skip=args.frame_skip,
      save_mode=args.save,
      display=args.display,
    )

  elif args.rtsp_file:
    try:
      with open(args.rtsp_file, "r") as f:
        rtsp_urls = [line.strip() for line in f if line.strip()
                     and not line.startswith("#")]
      print(f"Loaded {len(rtsp_urls)} RTSP streams from {args.rtsp_file}")
      manager = MultiStreamManager(detector)
      manager.process_multiple_streams(
        rtsp_urls=rtsp_urls,
        frame_skip=args.frame_skip,
        save_mode=args.save,
        display=args.display,
      )
    except FileNotFoundError:
      print(f"Error: File {args.rtsp_file} not found")
    except Exception as e:
      print(f"Error reading file {args.rtsp_file}: {e}")

  elif args.rtsp:
    processor = StreamProcessor(detector)
    processor.process_rtsp_stream(
      rtsp_url=args.rtsp,
      frame_skip=args.frame_skip,
      display=not args.no_display,
      save_mode=args.save,
    )

  else:
    print("No input specified. Use --help for usage information.")
    print("\nExample usage:")
    print("  Single stream:    python main.py --rtsp 'rtsp://camera1.com/stream' --save image")
    print("  Multiple streams: python main.py --rtsp-list 'rtsp://cam1.com' 'rtsp://cam2.com' --save video")
    print("  With display:     python main.py --rtsp-list 'rtsp://cam1.com' --save image --display")
    print("  From file:        python main.py --rtsp-file rtsp_streams.txt --save video --display")
    print("  Test image:       python main.py --test-image 'image.jpg' --save image")


if __name__ == "__main__":
  main()
