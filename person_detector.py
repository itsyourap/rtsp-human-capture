import cv2
import numpy as np
import threading
from typing import List, Tuple, Optional


class PersonDetector:
  """Loads a person-detection model (YOLOv4 / YOLOv3 / HOG fallback) and
  exposes a single thread-safe ``detect_persons`` method."""

  def __init__(
    self,
    confidence_threshold: float = 0.5,
    person_area_threshold: int = 1000,
    model_dir: str = "model",
  ) -> None:
    """Initialize person detector.

    Args:
        confidence_threshold: Minimum detection score (0–1).
        person_area_threshold: Minimum bounding-box area in pixels.
        model_dir: Directory that contains the YOLO weight/config files
                   and ``coco.names``.
    """
    print("Loading person detection model...")

    # Download model files from:
    # https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    # https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
    # https://github.com/AlexeyAB/darknet/blob/master/data/coco.names

    self.confidence_threshold: float = confidence_threshold
    self.person_area_threshold: int = person_area_threshold
    self.model_dir: str = model_dir
    self.net: Optional[cv2.dnn.Net] = None
    self.hog: Optional[cv2.HOGDescriptor] = None
    self.classes: List[str] = []
    self.layer_names: List[str] = []
    self.output_layers: List[str] = []
    self._inference_lock: threading.Lock = threading.Lock()

    # Try to load YOLO model files
    try:
      self.net = cv2.dnn.readNet(
        f"{model_dir}/yolov4.weights", f"{model_dir}/yolov4.cfg")
      model_name = "YOLOv4"
    except:
      try:
        self.net = cv2.dnn.readNet(
          f"{model_dir}/yolov3.weights", f"{model_dir}/yolov3.cfg")
        model_name = "YOLOv3"
      except:
        print("Warning: YOLO weights not found. Using OpenCV's built-in HOG person detector as fallback.")
        self.net = None
        self.hog = cv2.HOGDescriptor()
        # Convert to numpy array for setSVMDetector
        default_people_detector = np.array(
          cv2.HOGDescriptor.getDefaultPeopleDetector(), dtype=np.float32)
        self.hog.setSVMDetector(default_people_detector)
        model_name = "HOG"

    # Try to use NVIDIA GPU via CUDA backend
    if self.net is not None:
      cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
      if cuda_available:
          print("CUDA available, using GPU for inference")
          self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
      else:
          print("CUDA not available, using CPU for inference")
          self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
          self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load COCO class names
    try:
      with open(f"{model_dir}/coco.names", "r") as f:
        self.classes = [line.strip() for line in f.readlines()]
    except:
      # Default COCO classes if file not found
      self.classes = ["person", "bicycle", "car",
                      "motorbike", "aeroplane", "bus", "train", "truck"]

    if self.net is not None:
      self.layer_names = list(self.net.getLayerNames())
      unconnected_layers = self.net.getUnconnectedOutLayers()

      # Handle both numpy array and sequence types properly
      if isinstance(unconnected_layers, np.ndarray):
        # Handle numpy array case
        indices_list = unconnected_layers.flatten().tolist()
      else:
        # Handle list/sequence case
        indices_list = list(unconnected_layers)

      self.output_layers = [self.layer_names[i - 1] for i in indices_list]

    print(f"Model loaded: {model_name}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Person area threshold: {person_area_threshold} pixels")

  def detect_persons_yolo(self, frame: cv2.typing.MatLike) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect persons using YOLO model.
    Returns: list of bounding boxes [(x, y, w, h, confidence), ...]
    """
    try:
      height, width, channels = frame.shape

      # Create blob from image - ensure frame is not modified
      blob = cv2.dnn.blobFromImage(
        frame.copy(), 0.00392, (416, 416), (0, 0, 0), True, crop=False)
      if self.net is not None:
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
      else:
        return []

      boxes: List[List[int]] = []
      confidences: List[float] = []
      class_ids: List[int] = []

      for output in outputs:
        for detection in output:
          # Cast detection to numpy array for proper indexing
          detection_array = np.array(detection)
          scores = detection_array[5:]
          class_id = int(np.argmax(scores))
          confidence = float(scores[class_id])

          # Only detect persons (class_id = 0 in COCO)
          if class_id == 0 and confidence > self.confidence_threshold:
            center_x = int(detection_array[0] * width)
            center_y = int(detection_array[1] * height)
            w = int(detection_array[2] * width)
            h = int(detection_array[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Ensure bounding box is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            # Filter by area threshold and ensure positive dimensions
            if w > 0 and h > 0 and w * h > self.person_area_threshold:
              boxes.append([x, y, w, h])
              confidences.append(float(confidence))
              class_ids.append(class_id)

      # Apply Non-Maximum Suppression with more strict threshold
      person_boxes: List[Tuple[int, int, int, int, float]] = []
      if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
          boxes, confidences, self.confidence_threshold, 0.3)  # Reduced NMS threshold

        if len(indices) > 0:
          # Handle both numpy array and list cases for indices
          if isinstance(indices, np.ndarray):
            indices_flat = indices.flatten()
          else:
            indices_flat = indices
          for i in indices_flat:
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            person_boxes.append((x, y, w, h, confidence))

      return person_boxes

    except Exception as e:
      print(f"Error in YOLO detection: {e}")
      return []

  def detect_persons_hog(self, frame: cv2.typing.MatLike) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect persons using HOG descriptor (fallback method).
    Returns: list of bounding boxes [(x, y, w, h, confidence), ...]
    """
    try:
      if self.hog is None:
        return []

      # Resize frame for better performance
      height, width = frame.shape[:2]
      scale = min(640 / width, 480 / height)
      new_width, new_height = int(width * scale), int(height * scale)
      resized = cv2.resize(frame.copy(), (new_width, new_height))

      # Detect people
      boxes, weights = self.hog.detectMultiScale(
        resized, winStride=(8, 8), padding=(32, 32), scale=1.05)

      person_boxes: List[Tuple[int, int, int, int, float]] = []
      for i, (x, y, w, h) in enumerate(boxes):
        # HOG weights handling - weights can be different formats
        if len(weights) > i:
          weight_val = weights[i]
          if isinstance(weight_val, (list, tuple, np.ndarray)) and len(weight_val) > 0:
            confidence = float(weight_val[0])
          else:
            confidence = float(weight_val)
        else:
          confidence = 0.5

        # Scale back to original size
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        # Ensure bounding box is within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        # Filter by area and confidence
        if w > 0 and h > 0 and w * h > self.person_area_threshold and confidence > self.confidence_threshold:
          person_boxes.append((x, y, w, h, confidence))

      return person_boxes

    except Exception as e:
      print(f"Error in HOG detection: {e}")
      return []

  def detect_persons(self, frame: cv2.typing.MatLike) -> Tuple[bool, int, List[Tuple[int, int, int, int, float]]]:
    """
    Detect persons in frame using available method.
    Thread-safe: acquires an internal lock so that only one thread
    runs inference at a time (OpenCV DNN / HOG are not thread-safe).
    Returns: (has_person: bool, person_count: int, boxes: list)
    """
    with self._inference_lock:
      if self.net is not None:
        boxes = self.detect_persons_yolo(frame)
      else:
        boxes = self.detect_persons_hog(frame)

    has_person = len(boxes) > 0
    person_count = len(boxes)

    return has_person, person_count, boxes
