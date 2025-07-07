class SimpleIDAssigner:
    def __init__(self):
        """
        Simple counter to assign unique visitor IDs.
        """
        self.next_id = 1

    def get_new_id(self):
        """
        Return a new unique visitor ID.
        """
        new_id = f"visitor_{self.next_id}"
        self.next_id += 1
        return new_id
