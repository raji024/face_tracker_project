import insightface
import numpy as np

class FaceRecognizer:
    def __init__(self):
        """
        Load the InsightFace model for face embeddings.
        """
        print("[INFO] Loading InsightFace ArcFace model...")
        self.model = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, face_img):
        """
        Get 512D face embedding from cropped face image.
        Returns numpy array or None.
        """
        faces = self.model.get(face_img)
        if faces:
            return faces[0].embedding
        return None

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        Returns float between -1 and 1.
        """
        if vec1 is None or vec2 is None:
            return 0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
