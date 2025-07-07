import os
from datetime import datetime
import cv2

class EventLogger:
    def __init__(self, base_folder):
        """
        Initialize logger with base folder.
        """
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def save_event(self, face_id, face_img, event_label):
        """
        Save cropped face image with timestamp and ID label.
        Returns saved image path.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        folder = os.path.join(self.base_folder, event_label, date_str)
        os.makedirs(folder, exist_ok=True)

        filename = f"{face_id}_{time_str}.jpg"
        path = os.path.join(folder, filename)

        cv2.imwrite(path, face_img)
        return path
