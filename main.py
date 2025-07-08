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
# Similarity threshold (tune this as needed)
# ------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.55

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
# Store multiple embeddings per visitor
# ------------------------------------------------------------
known_embeddings = {}  # {ID: [embedding1, embedding2, ...]}

# ------------------------------------------------------------
# Track visitors already logged
# ------------------------------------------------------------
logged_visitors = set()

# ------------------------------------------------------------
# Open your new video file
# ------------------------------------------------------------
video_path = r"D:\projects\face_tracker_project\new_2.mp4"   # ✅ YOUR UPDATED PATH
cap = cv2.VideoCapture(video_path)
frame_count = 0

if not cap.isOpened():
    print(f"[ERROR] Cannot open video file: {video_path}")
    exit(1)

print("[INFO] Starting face tracking...")

# ------------------------------------------------------------
# Main video processing loop
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream reached.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # --------------------------------------------------------
    # Detect faces in this frame
    # --------------------------------------------------------
    boxes = detector.detect_faces(frame)
    print(f"[DEBUG] Detected {len(boxes)} faces in frame {frame_count}")

    for box in boxes:
        x1, y1, x2, y2 = box

        # ⭐ 1. Clip box to image size
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        # ⭐ 2. Crop face region safely
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            print(f"[WARNING] Empty face crop (box={box}). Skipping.")
            continue

        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            print(f"[WARNING] Face crop too small (shape={face_img.shape}). Skipping.")
            continue

        # ⭐ 3. Convert BGR to RGB
        try:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] Color conversion failed: {e}")
            continue

        # ⭐ 4. Get embedding with error handling
        try:
            embedding = recognizer.get_embedding(face_img_rgb)
            if embedding is None:
                print("[WARNING] Could not extract embedding for face (model returned None).")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to get embedding: {e}")
            continue

        # ⭐ 5. Normalize embedding
        norm = (embedding ** 2).sum() ** 0.5
        if norm == 0:
            print("[WARNING] Zero-norm embedding. Skipping.")
            continue
        embedding = embedding / norm

        # ----------------------------------------------------
        # Match with existing visitors
        # ----------------------------------------------------
        matched_id = None
        best_similarity = -1.0

        for face_id, embeddings_list in known_embeddings.items():
            similarities = [recognizer.cosine_similarity(embedding, e) for e in embeddings_list]
            max_similarity = max(similarities)
            print(f"[DEBUG] Max similarity with ID {face_id}: {max_similarity:.3f}")

            if max_similarity > SIMILARITY_THRESHOLD and max_similarity > best_similarity:
                matched_id = face_id
                best_similarity = max_similarity

        # ----------------------------------------------------
        # If new visitor, register
        # ----------------------------------------------------
        if matched_id is None:
            matched_id = id_assigner.get_new_id()
            known_embeddings[matched_id] = [embedding]
            print(f"[INFO] New visitor registered: {matched_id}")
        else:
            print(f"[INFO] Recognized existing visitor: {matched_id} (similarity={best_similarity:.3f})")
            # ⭐ Add new embedding to improve future matching
            known_embeddings[matched_id].append(embedding)

        # ----------------------------------------------------
        # Log ENTRY event only once per visitor
        # ----------------------------------------------------
        if matched_id not in logged_visitors:
            timestamp = datetime.now().isoformat()
            image_path = logger.save_event(matched_id, face_img, "ENTRY")
            database.log_event(matched_id, timestamp, "ENTRY", image_path)
            logged_visitors.add(matched_id)
            print(f"[INFO] Logged ENTRY for {matched_id}")
        else:
            print(f"[INFO] Already logged {matched_id}, skipping saving.")

# ------------------------------------------------------------
# Clean up resources
# ------------------------------------------------------------
cap.release()
try:
    cv2.destroyAllWindows()
except:
    pass

print("[INFO] Video processing complete.")
print(f"[RESULT] Total distinct visitors detected: {len(known_embeddings)}")
