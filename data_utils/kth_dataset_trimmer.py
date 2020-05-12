import cv2
import os
import pickle
path = './data/kth-actions/00sequences.txt'


import pickle

import os
from tqdm import tqdm


def trim_vid(dict=None, root_path=None, input_path=None, output_path=None):
    '''
    Trims a video file in the KTH-datasets and splits it into multiple videos where the 
    dead frames (where no person is in shot) are cutout. 

    TODO: Concatination function if necessary

    Args:
        dict (dict): a dictionary containing the data in the following form
                        {'file_name': person07_jogging_d2,
                         'action_name': 'jogging',
                         'frames_from': [1,190,235,382],
                         'frames_to': [159,210,297,400],
                         } 
        root_path   (str): the root folder of the kth-datasets eg. './data/kth-actions',
        input_path  (str): the folder where the kth-dataset videos are eg. './data/kth-actions/video',
        output_path (str): the folder where the trimmed videos will be exported eg './data/kth-actions/trimmed',

    Returns:
        void  
    '''
    FPS = 25.

    filename = dict['file_name']
    start = dict['frames_from']
    action_name = dict['action_name']
    end = dict['frames_to']

    tmp_path = os.path.join(root_path, 'tmp')
    input_root_path = os.path.join(input_path, action_name)
    input_file_path = os.path.join(input_root_path, filename)
    output_root_path = os.path.join(output_path, action_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    if not os.path.exists(output_root_path):
        os.mkdir(output_root_path)

    start = list(map(lambda x: x / FPS, start))
    end = list(map(lambda x: x / FPS, end))

    save_file_paths = []
    for s, e in zip(start, end):
        output_file_path = os.path.join(
            output_root_path, f"{filename}_{dashed(s,FPS)}_to_{dashed(e,FPS)}")
        save_file_paths.append(output_file_path)
        cmd = f'ffmpeg -i {input_file_path}_uncomp.avi -ss {s} -t {e-s} -c:v libx264 -c:a aac {output_file_path}.avi'
        os.system(cmd)


def dashed(s, fps):
    '''
    converts a string in seconds to frames and replaces all dots with undescores.
    Eg "file_name_from_2.3_to_3.2" becoems "file_name_from_57_to_80"
    args:
        s (str): a string of the 
    '''
    return str(int(s * fps)).replace('.', '_')


def concat(saving_paths, filename, output_path):
    '''
    No used anywhere yet!
    '''
    with open('tmp/out.txt', 'w') as f:
        for item in saving_paths:
            f.write("file '%s'\n" % item)

    cmd = "ffmpeg -f concat -i tmp/out.txt -c copy {}/{}_trimmed.avi".format(
        output_path, filename)
    os.system(cmd)
    # if concat == True:
    #     with open(os.path.join(tmp_path, 'out.txt'), 'w') as file:
    #         for path in file:
    #             # do concat
    #             pass
    # # # concatinate videos
    # # concat(saving_paths, filename, output_path)
    # # os.system("rm -r tmp/*")


def export_frames_info(path):
    frames_info = dict()
    '''
    Reads the data inside 00sequences.txt and extract the relevant sequences
    (no empty scenes) in each video. The extracted data is pickled and
    saved as dictionary in the following form:

    {file_name: {
        'from': [seqeunce1_start_frame, seqeunce2_start_frame, seqeunce3_start_frame],
        'to': [seqeunce1_end_frame, seqeunce2_end_frame, seqeunce3_end_frame]
    }}
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split()
            file_name = line[0]
            if len(line) == 6:
                sequences = line[2:6]
            elif len(line) == 5:
                sequences = line[2:5]
            else:
                pass
            frames_info[file_name] = {'from': [], 'to': []}
            for j, seq in enumerate(sequences):
                seq_info = seq.split('-')
                frames_info[file_name]['from'].append(int(seq_info[0]))
                frames_info[file_name]['to'].append(
                    int(seq_info[1].split(',')[0]))

    with open('./data/kth-actions/frames_info.pkl', 'wb') as target:
        pickle.dump(frames_info, target)


if __name__ == "__main__":
    SHOW_EXAMPLE = True
    with open('./data/kth-actions/frames_info.pkl', 'rb') as target:
        frames_info = pickle.load(target)
    if SHOW_EXAMPLE:
        print(frames_info['person15_boxing_d4'])
        print(frames_info['person14_walking_d2'])

    # Convert from frames_info to info that can be used by the trim_vid function
    llist = []
    for k, v in frames_info.items():
        d = {}
        d['file_name'] = k
        d['action_name'] = k.split('_')[1]
        d['frames_from'] = v['from']
        d['frames_to'] = v['to']
        llist.append(d)

    for elem in llist:
        trim_vid(elem,
                 root_path='./data/kth-actions',
                 input_path='./data/kth-actions/video',
                 output_path='./data/kth-actions/trimmed')
