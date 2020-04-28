"""Module for loading labels.

Example:
    label_loader = LabelLoader()
    labels = label_loader.load_labels()
    label_dict = label_loader.get_label_dict()
    
"""

import json


class LabelLoader:
    """Used to load videos as numpy arrays."""

    def __init__(self, label_folder_path='./data/something-something-mini-anno'):
        """Constructor."""
        self.label_folder_path = label_folder_path

        self.labels_file_path = "{}/something-something-v2-labels.json".format(self.label_folder_path)
        self.train_file_path = "{}/train_videofolder.txt".format(self.label_folder_path)
        self.valid_file_path = "{}/val_videofolder.txt".format(self.label_folder_path)
        self.test_file_path = "{}/test_videofolder.txt".format(self.label_folder_path)

    def load_labels(self):
        """Load labels.

        Returns:
            Dict with keys "train", "valid", "test". Each key maps to a list of dicts, with each
            dict corresponding to a single sample. Each sample dict has attributes "id", "n_frames"
            "label", and "setname".

        """
        setnames = ["train", "valid", "test"]
        paths = [self.train_file_path, self.valid_file_path, self.test_file_path]

        labels = {}
        for setname, path in zip(setnames, paths):
            with open(path, mode='r') as f:
                valdicts = []
                for line in f:
                    values = line.split()
                    valdict = {'id': values[0],
                               'n_frames': values[1],
                               'label': values[2],
                               'setname': setname}
                    valdicts.append(valdict)
                labels[setname] = valdicts

        return labels

    def get_label_dict(self):
        """Return a dictionary mapping label indices to label descriptions."""
        with open(self.labels_file_path, mode='r') as f:
            label_file = f.read()

        inverse_label_dict = json.loads(label_file)
        label_dict = {int(value): key for key, value in inverse_label_dict.items()}
        return label_dict


if __name__ == "__main__":
    label_loader = LabelLoader()
    labels = label_loader.load_labels()
    label_dict = label_loader.get_label_dict()
