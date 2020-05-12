import cv2
import os

path = '../data/kth-actions/00sequences.txt'
frames_info = dict()
'''
{file_name: {
    'from': [seqeunce1_start, seqeunce2_start, seqeunce3_start],
    'to': [seqeunce1_end, seqeunce2_end, seqeunce3_end]
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
            frames_info[file_name]['to'].append(int(seq_info[1].split(',')[0]))
print(frames_info['person14_walking_d2'])
print(frames_info['person15_boxing_d4'])
