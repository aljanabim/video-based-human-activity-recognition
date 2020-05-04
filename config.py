

ROOT_PATH = "../data/something-something-mini"
class Config():
    def __init__(self):
        self.root_path = ROOT_PATH
        self.anno_path = "{}-anno".format(ROOT_PATH)
        self.frame_path = "{}-frame".format(ROOT_PATH)

        self.n_classes=174
        self.img_width=455
        self.img_height=256