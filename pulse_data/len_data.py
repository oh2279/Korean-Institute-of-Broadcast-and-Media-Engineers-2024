'''
pulse_data 폴더 내의 모든 txt 파일의 길이를 출력
'''

import os

data_dir = '/home/gpuadmin/papers/beomseok/Broadcast_and_Media/augmentation/pulse_data/pulse_raw'

for file in os.listdir(data_dir):
    if file.endswith('.txt'):
        with open(os.path.join(data_dir, file), 'r') as f:
            len_data = len(f.readlines())
            if len_data < 40000:
                print("file name: ", file, "length: ", len_data)