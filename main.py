import cv2
import json
from datetime import datetime

from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import SimpleIDAssigner
from logger import EventLogger
from database import VisitorDatabase

# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

FRAME_SKIP = config["frame_skip"]
LOG_FOLDER = config["log_folder"]
DB_FILE = config["database_file"]
YOLO_MODEL_PATH = config["yolo_model_path"]

# ------------------------------------------------------------
# Initialize components
# ------------------------------------------------------------
print("[INFO] Initializing modules...")
detector = FaceDetector(YOLO_MODEL_PATH)
recognizer = FaceRecognizer()
id_assigner = SimpleIDAssigner()
logger = EventLogger(LOG_FOLDER)
database = VisitorDatabase(DB_FILE)

# ------------------------------------------------------------
# Store known embeddings (FaceID -> Embedding)
# ------------------------------------------------------------
known_embeddings = {}

# ------------------------------------------------------------
# Open your specific video file
# ------------------------------------------------------------
cap = cv2.VideoCapture(
    r"D:\projects\face_tracker_project\4625295-hd_1920_1080_25fps.mp4"
)
frame_count = 0

print("[INFO] Starting face tracking...")

# ------------------------------------------------------------
# Main video processing loop
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream reached. Exiting loop.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # --------------------------------------------------------
    # Detect faces in this frame
    # --------------------------------------------------------
    boxes = detector.detect_faces(frame)
    for box in boxes:
        x1, y1, x2, y2 = box
        face_img = frame[y1:y2, x1:x2]

        # ----------------------------------------------------
        # Generate embedding for this face
        # ----------------------------------------------------
        embedding = recognizer.get_embedding(face_img)
        if embedding is None:
            continue

        # ----------------------------------------------------
        # Match with known visitors
        # ----------------------------------------------------
        matched_id = None
        for face_id, stored_embedding in known_embeddings.items():
            similarity = recognizer.cosine_similarity(embedding, stored_embedding)
            if similarity > 0.6:
                matched_id = face_id
                break

        # ----------------------------------------------------
        # If new visitor, register
        # ----------------------------------------------------
        if matched_id is None:
            matched_id = id_assigner.get_new_id()
            known_embeddings[matched_id] = embedding
            print(f"[INFO] New visitor registered: {matched_id}")

        # ----------------------------------------------------
        # Log ENTRY event
        # ----------------------------------------------------
        timestamp = datetime.now().isoformat()
        image_path = logger.save_event(matched_id, face_img, "ENTRY")
        database.log_event(matched_id, timestamp, "ENTRY", image_path)

# ------------------------------------------------------------
# Clean up resources
# ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------
# Print final result
# ------------------------------------------------------------
print("[INFO] Video processing complete.")
print(f"[RESULT] Total distinct visitors detected: {len(known_embeddings)}")
