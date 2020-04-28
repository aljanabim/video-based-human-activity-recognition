"""Module for loading labels.

Example:
    label_loader = LabelLoader()
    label_loader.load_files()

"""


class LabelLoader:
    """Used to load videos as numpy arrays."""

    def __init__(self, dataset_root='./data/something-something-mini-anno'):
        """Constructor."""
        self.dataset_root = dataset_root

        self.train_file_path = "{}/train_videofolder.txt".format(self.dataset_root)
        self.valid_file_path = "{}/val_videofolder.txt".format(self.dataset_root)
        self.test_file_path = "{}/test_videofolder.txt".format(self.dataset_root)

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
