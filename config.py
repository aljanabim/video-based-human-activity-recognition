import os

# ROOT_PATH = "./data/20bn-something-something-v2"  #FULL DATASET
ROOT_PATH = "./data/something-something-mini"
JSON_PATH = './data/20bn-something-something-v2-jason'
RECORDS_PATH = './data/records/'
class Config():
    def __init__(self):
        self.root_path = os.path.expanduser(ROOT_PATH)
        
        # videos to be used
        self.videos_path = os.path.expanduser("{}-video".format(ROOT_PATH))
        # annonations of the used videos
        self.jason_label_path = JSON_PATH

        ## output folders
        self.label_path =   os.path.expanduser("{}-label".format(ROOT_PATH))
        self.frame_path =  os.path.expanduser("{}-frame".format(ROOT_PATH))
        
                

        self.n_classes=174
        self.img_width=455
        self.img_height=256
        self.max_frames=70


        #video to frames
        self.n_frames = 40
        self.n_threads=100
        self.decode_video = True
        self.build_file_list = True


        # tf records
        self.record_output = RECORDS_PATH
