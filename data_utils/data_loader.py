"""Module for loading the data.

Example:
    from data_utils.data_loader import load_data()
    data, label_dict = load_data("../data/something-something-mini")"""
try:
    from data_utils.metadata_loader import MetadataLoader
    from data_utils.video_loader import VideoLoader
except ModuleNotFoundError:
    from metadata_loader import MetadataLoader
    from video_loader import VideoLoader


def load_data(root_path):
    """Return object containing all data contained in folders with the given root.

    Returns:
        data: Data object which has the below format:
              data = {
                      'train': {
                                id1: {'id': id1, 'label': int, 'video': Tensor(n, w, h, c)}
                                id2: {'id': id2, 'label': int, 'video': Tensor(n, w, h, c)}
                                ...
                      }
                      'valid': { -||- }
                      'test': { -||- }
              }
              where n is the number of frames in the video, w is the video width, h is the video height, and
              c is the number of channels.

        label_dict: dictionary mapping label indices to label descriptions.
    """
    frame_path = "{}-frame".format(root_path)
    anno_path = "{}-anno".format(root_path)

    video_loader = VideoLoader(img_width=455, img_height=256, video_folder_path=frame_path)
    videos = video_loader.load_all_videos()
    metadata_loader = MetadataLoader(label_folder_path=anno_path)
    metadata = metadata_loader.load_metadata()
    label_dict = metadata_loader.get_label_dict()

    data = {}
    for subset in metadata:
        data[subset] = {}
        subset_metadata = metadata[subset]
        for id in subset_metadata:
            sample_metadata = subset_metadata[id]
            data[subset][id] = {'id': id,
                                'action_label': sample_metadata['action_label'],
                                'data': videos[id]}
    return data, label_dict

if __name__ == "__main__":
    data, label_dict = load_data("../data/something-something-mini")
    data['train']
