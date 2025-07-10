An AI-powered visitor management solution that automatically detects, recognizes, and logs faces from video footage—built as our submission for the Katomaran Hackathon.

🚀 Features
✅ YOLO-based face detection
✅ Face recognition with embedding-based matching
✅ Multiple embeddings per visitor for pose variation robustness
✅ Unique visitor ID assignment
✅ Automatic event logging with timestamp and cropped images
✅ Configurable video input and detection thresholds
✅ Designed for offline, privacy-friendly use

🏗️ Architecture Overview
1️⃣ Detection: YOLO model identifies face locations in video frames.
2️⃣ Recognition: Embedding extractor encodes facial features.
3️⃣ Similarity Matching: Cosine similarity with known embeddings for ID assignment.
4️⃣ Tracking: Supports multiple embeddings per visitor to recognize them from different angles.
5️⃣ Logging: Saves visitor ENTRY events with timestamp and face crop for auditing.

📂 Project Structure
bash
Copy
Edit
.
├── detector.py          # YOLO-based face detection module
├── recognizer.py        # Embedding extractor and similarity computation
├── tracker.py           # Unique ID assignment
├── logger.py            # Image saving with folder organization
├── database.py          # Event database logging
├── config.json          # Configurable settings (paths, thresholds)
└── main.py              # Main video processing script
⚙️ How It Works
Loads a video file (e.g., CCTV footage).

Skips frames for speed (configurable).

Detects all faces in each frame.

Crops and preprocesses faces for embedding extraction.

Compares embeddings to known visitor records.

Assigns new IDs if no match found.

Logs first ENTRY event per visitor with image and timestamp.

🛠️ Setup
1️⃣ Install Requirements
bash
Copy
Edit
pip install opencv-python numpy
# Add other dependencies (torch, ultralytics, etc.) if needed
2️⃣ Add YOLO Model Weights
Place your YOLO weights file (e.g., yolov8n-face.pt) in the path specified in config.json.

Example config.json:

json
Copy
Edit
{
    "frame_skip": 5,
    "log_folder": "logs",
    "database_file": "visitors.json",
    "yolo_model_path": "yolov8n-face.pt"
}
3️⃣ Prepare Input Video
Place your video at the path you want to analyze, e.g.:

makefile
Copy
Edit
D:\projects\face_tracker_project\new_2.mp4
▶️ Run the System
bash
Copy
Edit
python main.py
✅ Logs cropped face images in your logs folder
✅ Stores visitor event records in your database file
✅ Console prints detection, recognition, and logging info

🧪 Example Use Cases
Office visitor attendance automation

Retail customer footfall analytics

Smart surveillance for secure facilities

Event access control and audit logs

💻 Katomaran Hackathon Focus
This project was developed specifically for the Katomaran Hackathon, addressing the challenge of automating visitor tracking and logging with computer vision.

Key Hackathon highlights:

Modular, easy-to-extend code

Works offline for privacy

Adaptable for different camera setups

Can be integrated with dashboards, alerts, or access control systems

📸 Example Outputs
Folder structure for logs:

Copy
Edit
logs/
└── visitor_1/
    └── entry_2024-07-06T14-23-55.jpg
Example visitors.json:

json
Copy
Edit
[
    {
        "id": 1,
        "timestamp": "2024-07-06T14:23:55",
        "event": "ENTRY",
        "image_path": "logs/visitor_1/entry_2024-07-06T14-23-55.jpg"
    }
]


📜 License
This project is provided for educational and Katomaran Hackathon demonstration purposes.

🌟 Acknowledgements
Ultralytics YOLOv8 for detection

OpenCV for video processing

Face recognition embeddings (ArcFace / custom models)

📬 Contact
For questions, contact: [rajeshwariak7@gmail.com]