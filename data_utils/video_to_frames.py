"""This script is for preprocessing something-something-v2 dataset.
"""

import argparse
import json
import os
import sys
import threading

sys.path.append('../')
sys.path.append('.')
from config import Config

def split_func(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def decode_video(config):
    if not os.path.exists(config.videos_path):
        raise ValueError('Please download videos and set video_root variable.')
    if not os.path.exists(config.frame_path):
        os.makedirs(config.frame_path)
    if not os.path.exists(config.label_path):
        os.makedirs(config.label_path)

    video_list = os.listdir(config.videos_path)
    # print(video_list)
    splits = list(split_func(video_list, config.n_threads))

    # sub_functions for extraction
    resolution_string = "{}x{}".format(config.img_width, config.img_height)

    def extract(video, tmpl='%06d.jpg'):
        cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -s {} -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(
            config.videos_path, video, resolution_string, config.frame_path, video[:-5])
        os.system(cmd)

    def target(video_list):
        for video in video_list:
            if os.path.exists(os.path.join(config.frame_path, video[:-5])):
                return
            os.makedirs(os.path.join(config.frame_path, video[:-5]))
            extract(video)

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def build_file_list(config):
    n_deleted_folders = 0

    if not os.path.exists(config.jason_label_path):
        raise ValueError(
            'Please download annotations and set label_path variable.')

    dataset_name = 'something-something-v2'
    with open(os.path.join(config.jason_label_path, '%s-labels.json' % dataset_name)) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    # with open('category.txt', 'w') as f:
    #     f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [os.path.join(config.jason_label_path, '%s-validation.json' % dataset_name),
                   os.path.join(config.jason_label_path,
                                '%s-train.json' % dataset_name),
                   os.path.join(config.jason_label_path, '%s-test.json' % dataset_name)]

    files_output = [os.path.join(config.label_path, 'val_videofolder.txt'),
                    os.path.join(config.label_path, 'train_videofolder.txt'),
                    os.path.join(config.label_path, 'test_videofolder.txt')]

    for (filename_input, filename_output) in zip(files_input, files_output):
        if os.path.exists(filename_output):
            continue
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(
                    dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            # Ugly way of handling missing files, allows using smaller sets /Einar
            try:
                curFolder = folders[i]
                curIDX = idx_categories[i]
                # counting the number of frames in each video folders
                dir_files = os.listdir(os.path.join(
                    config.frame_path, curFolder))
                if len(dir_files) == 0:
                    print('WARNING: Error when building file list, frame folder empty at %s, deleting folder' % (
                        curFolder))
                    os.rmdir(os.path.join(config.frame_path, curFolder))
                    n_deleted_folders += 1
                    # sys.exit()
                else:
                    output.append('%s %d %d' %
                                  (curFolder, len(dir_files), curIDX))
                    print('%d/%d' % (i, len(folders)))
            except FileNotFoundError:
                pass
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

    print("Deleted folders: {}".format(n_deleted_folders))


def decode_videos(config):
    """Decode videos stored in video format to folders containing jpgs.
    """

    if config.decode_video:
        print('Decoding videos to frames.')
        decode_video(config)
        print(config.videos_path)

    if config.build_file_list:
        print('Generating training files.')
        build_file_list(config)
        print(config.label_path)


if __name__ == '__main__':
    config = Config(use_subfolders=False)
    decode_videos(config)
