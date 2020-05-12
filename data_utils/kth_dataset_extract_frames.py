
import pickle

import os
from tqdm import tqdm


def trim_vid(dict=None, root_path=None, input_path=None, output_path=None, output_dir='trimmed'):
    FPS = 25.
    filename = dict['file_name']
    start = dict['frames_from']
    action_name = dict['action_name']
    end = dict['frames_to']

    tmp_path = os.path.join(root_path, 'tmp')
    input_root_path = os.path.join(input_path, action_name)
    input_file_path = os.path.join(input_root_path, filename)
    output_root_path = os.path.join(output_path, action_name)
    output_file_path = os.path.join(output_root_path, filename)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    if not os.path.exists(output_root_path):
        os.mkdir(output_root_path)

    # input_path = input_path + "/video/" + action_name
    # # filename = "aa"
    start = list(map(lambda x: x / FPS, start))
    end = list(map(lambda x: x / FPS, end))

    # saving_paths = []
    # print(os.listdir(input_path))

    # cmd = "ffmpeg -i {}.avi".format(file_path)
    # print(file_path)
    # print(output_path)

    save_file_paths = []
    for s, e in zip(start, end):
        tmp_file_path = os.path.join(
            tmp_path, f"out_{dashed(s)}_to_{dashed(e)}")
        save_file_paths.append(tmp_file_path)
        # saving_paths.append("name{}.avi".format(str(s).replace('.', '_')))
        # cmd = cmd + " -t {}  tmp/{} -ss {}".format(e - s, saving_paths[-1], s)
        # cmd = cmd + " -t {}  tmp/{} -ss {}".format(e - s, saving_paths[-1], s)
        cmd = f'ffmpeg -i {input_file_path}_uncomp.avi -ss {s} -t {e-s} -c:v libx264 -c:a aac {output_file_path}.avi'
        os.system(cmd)

    if concat == True:
        with open(os.path.join(tmp_path, 'out.txt'), 'w') as file:
            for path in file:
                # do concat
                pass
    # # concatinate videos
    # concat(saving_paths, filename, output_path)
    # os.system("rm -r tmp/*")


def dashed(s):
    return str(int(s * 25)).replace('.', '_')


def concat(saving_paths, filename, output_path):
    with open('tmp/out.txt', 'w') as f:
        for item in saving_paths:
            f.write("file '%s'\n" % item)

    cmd = "ffmpeg -f concat -i tmp/out.txt -c copy {}/{}_trimmed.avi".format(
        output_path, filename)
    os.system(cmd)


# def trimm_all(dictionary_list, input_path):
#     for d in dictionary_list:
#         trim_vid(d, input_path)


if __name__ == "__main__":
    with open('./data/kth-actions/frames_info.pkl', 'rb') as target:
        frames_info = pickle.load(target)
    action_names = os.listdir('./data/kth-actions/video')

    # print(action_names)
    llist = []
    for k, v in frames_info.items():
        d = {}
        d['file_name'] = k
        d['action_name'] = k.split('_')[1]
        d['frames_from'] = v['from']
        d['frames_to'] = v['to']
        llist.append(d)

        # print(file_name, frames_from, frames_to)
    # print(llist[0])
    trim_vid(llist[0],
             root_path='./data/kth-actions',
             input_path='./data/kth-actions/video',
             output_path='./data/kth-actions/trimmed',
             output_dir='trimmed')
    # trimm_all(llist, "./data/kth-actions")
    # print(frames_info['person14_walking_d2'])
    # print(frames_info['person15_boxing_d4'])

    # d = {}
    # d['filename']
    # d['from']
    # d['to']

    # llist = [d1, d2, d3, d4]
