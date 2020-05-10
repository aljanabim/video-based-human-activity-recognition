"""Module for loading metadata.
    metadata_loader = MetadataLoader()
    metadata = metadata_loader.load_metadata()
    label_dict = metadata_loader.get_label_dict()
"""
import sys
sys.path.append('.')
sys.path.append('../')

from config import Config

import json


class MetadataLoader:
    """Used to load video metadata."""

    def __init__(self, config):
        """Constructor."""
        self.labels_file_path = "{}/something-something-v2-labels.json".format(
            config.jason_label_path)
        self.train_file_path = "{}/train_videofolder.txt".format(
            config.label_path)
        self.valid_file_path = "{}/val_videofolder.txt".format(
            config.label_path)
        self.test_file_path = "{}/test_videofolder.txt".format(
            config.label_path)

    def load_metadata(self):
        """Load labels.

        Returns:
            Dict with keys "train", "valid", "test". Each key maps to a dicts, with each
            keys being sample ids. Each key maps to a dict with keys "n_frames",
            "label", and "setname".

        """
        set_names = ["train", "valid", "test"]
        paths = [self.train_file_path,
                 self.valid_file_path, self.test_file_path]

        metadata = {}
        for set_name, path in zip(set_names, paths):
            with open(path, mode='r') as f:
                subset_metadata = {}
                for line in f:
                    values = line.split()
                    id = int(values[0])
                    valdict = {'n_frames': values[1],
                               'action_label': int(values[2]),
                               'object_labels': None,
                               'set_name': set_name}
                    subset_metadata[id] = valdict
                metadata[set_name] = subset_metadata

        return metadata

    def get_label_dict(self):
        """Return a dictionary mapping label indices to label descriptions."""
        with open(self.labels_file_path, mode='r') as f:
            label_file = f.read()

        inverse_label_dict = json.loads(label_file)
        label_dict = {int(value): key for key,
                      value in inverse_label_dict.items()}
        return label_dict


if __name__ == "__main__":

    config = Config(root_path='./data/1000-videos', use_subfolders=True)
    metadata_loader = MetadataLoader(config)
    metadata = metadata_loader.load_metadata()
    label_dict = metadata_loader.get_label_dict()
