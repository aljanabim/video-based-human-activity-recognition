"""This script is for preprocessing something-something-v2 dataset.

The code is largely borrowed from https://github.com/MIT-HAN-LAB/temporal-shift-module
and https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
"""

####
"""
python3 somethingsomethingv2.py --video_root=../../data/something-something-mini-video
                                --frame_root=../../data/something-something-mini-fram2
                                --anno_root=../../data/something-something-mini-anno2
"""
import sys
sys.path.append('../')
from config import Config

import os
import threading
import argparse
import json


def split_func(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(args.frame_root, video[:-5]))
        extract(video)


def decode_video(args, target_function):
    print(args.video_root)
    if not os.path.exists(args.video_root):
        raise ValueError('Please download videos and set video_root variable.')
    if not os.path.exists(args.frame_root):
        os.makedirs(args.frame_root)
    if not os.path.exists(args.label_root):
        os.makedirs(args.label_root)

    video_list = os.listdir(args.video_root)
    splits = list(split_func(video_list, args.num_threads))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target_function, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def build_file_list(args):
    if not os.path.exists(args.anno_root):
        raise ValueError('Please download annotations and set anno_root variable.')

    dataset_name = 'something-something-v2'
    with open(os.path.join(args.anno_root, '%s-labels.json' % dataset_name)) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [os.path.join(args.anno_root, '%s-validation.json' % dataset_name),
                   os.path.join(args.anno_root, '%s-train.json' % dataset_name),
                   os.path.join(args.anno_root, '%s-test.json' % dataset_name)]
    files_output = [os.path.join(args.label_root, 'val_videofolder.txt'),
                    os.path.join(args.label_root, 'train_videofolder.txt'),
                    os.path.join(args.label_root, 'test_videofolder.txt')]
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            # Ugly way of handling missing files, allows using smaller sets /Einar
            try:
                curFolder = folders[i]
                curIDX = idx_categories[i]
                # counting the number of frames in each video folders
                dir_files = os.listdir(os.path.join(args.frame_root, curFolder))
                if len(dir_files) == 0:
                    print('video decoding fails at %s' % (curFolder))
                    sys.exit()
                output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
                print('%d/%d' % (i, len(folders)))
            except FileNotFoundError:
                pass
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))


def make_arg_object(videos_path, json_path, output_path, n_frames, n_threads,
                    img_width, img_height):
    class ArgObject:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    frames_path = output_path + '/frames'
    label_path = output_path + '/labels'

    args = ArgObject(video_root=os.path.expanduser(videos_path),
                     anno_root=os.path.expanduser(json_path),
                     frame_root=os.path.expanduser(frames_path),
                     label_root=os.path.expanduser(label_path),
                     frame_nr=n_frames,
                     num_threads=n_threads,
                     decode_video=True,
                     build_file_list=True,
                     img_width=img_width,
                     img_height=img_height)

    return args


def decode_videos(videos_path, json_path, output_path, n_frames=40, n_threads=100,
                  img_width=84, img_height=84):
    """Decode videos stored in video format to folders containing jpgs.

    Args:
        videos_path: Path of directory containing video files.
        json_path: Path of directory containing json data files.
        output_path: Root of directory that will contain ouput frames and labels.
        n_frames: (Approximately?) number of output frames per video.
        n_threads: No clue what this does really.
        img_width: Width in pixels of output frames.
        img_height: Height in pixels of output frames.

    """
    args = make_arg_object(videos_path, json_path, output_path, n_frames, n_threads,
                           img_width, img_height)

    resolution_string = "{}x{}".format(args.img_width, args.img_height)

    def extract(video, tmpl='%06d.jpg'):
        cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -s {} -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(
            args.video_root, video, resolution_string, args.frame_root, video[:-5])
        os.system(cmd)

    def target(video_list):
        for video in video_list:
            os.makedirs(os.path.join(args.frame_root, video[:-5]))
            extract(video)


    if args.decode_video:
        print('Decoding videos to frames.')
        decode_video(args, target_function=target)

    if args.build_file_list:
        print('Generating training files.')
        build_file_list(args)


if __name__ == '__main__':
    videos_path = './data/something-something-mini-video'
    json_path = './data/something-something-mini-anno'
    output_path = './data/testdata'

    decode_videos(videos_path, json_path, output_path)
