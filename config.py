import os

# ROOT_PATH = "./data/20bn-something-something-v2"  #FULL DATASET
ROOT_PATH = "./data/something-something-mini"
JSON_PATH = './data/20bn-something-something-v2-jason'
RECORDS_PATH = './data/records/'
class Config():
    def __init__(self, root_path=ROOT_PATH, json_path=JSON_PATH, n_classes=174,
                 img_width=455, img_height=256, max_frames=70, n_frames=40, use_subfolders=False):

        separator = '/' if use_subfolders else '-'

        self.root_path = os.path.expanduser(root_path)

        # videos to be used
        self.videos_path = os.path.expanduser("{}{}video".format(root_path, separator))
        # annonations of the used videos
        self.jason_label_path = json_path

        ## output folders
        self.label_path = os.path.expanduser("{}{}label".format(root_path, separator))
        self.frame_path = os.path.expanduser("{}{}frame".format(root_path, separator))

        self.n_classes = n_classes
        self.img_width = img_width
        self.img_height = img_height
        self.max_frames = max_frames

        #video to frames
        self.n_frames = n_frames
        self.n_threads = 100
        self.decode_video = True
        self.build_file_list = True

        # tf records
        self.record_output = RECORDS_PATH
