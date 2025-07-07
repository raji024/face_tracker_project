import sqlite3

class VisitorDatabase:
    def __init__(self, db_file):
        """
        Connect to SQLite database.
        """
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        """
        Create visitors table if not exists.
        """
        query = """
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            timestamp TEXT,
            event TEXT,
            image_path TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def log_event(self, face_id, timestamp, event, image_path):
        """
        Insert visitor event into database.
        """
        query = """
        INSERT INTO visitors (face_id, timestamp, event, image_path)
        VALUES (?, ?, ?, ?)
        """
        self.conn.execute(query, (face_id, timestamp, event, image_path))
        self.conn.commit()

    def get_unique_visitors(self):
        """
        Retrieve list of unique visitor IDs.
        """
        query = "SELECT DISTINCT face_id FROM visitors"
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]
