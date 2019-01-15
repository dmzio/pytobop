import json


class BaseLogger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self, config=None):
        self.entries = {}

    def add_entry(self, entry: dict):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

    def watch(self, model):
        """
        watches model state
        """
        pass
