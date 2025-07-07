from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path):
        """
        Initialize and load the YOLO model.
        """
        print(f"[INFO] Initializing YOLO model: {model_path}")
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        """
        Detect faces in a frame using YOLO.
        Returns list of (x1, y1, x2, y2) boxes.
        """
        results = self.model(frame)
        boxes = []
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))
        return boxes
