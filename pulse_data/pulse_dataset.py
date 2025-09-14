import os
import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader


class PulseDataset(Dataset):
    def __init__(self, data_dir, label_excel_path, channel=5, flat_interval=(7000, 43000), transform=None):
        """
        data_dir: txt파일이 있는 폴더 경로
        label_excel_path: label 엑셀 파일 경로
        channel: 사용할 채널의 수, 1일 경우 ch3, 5일 경우 ch1, ch2, ch3, ch4, ch5 모두
        flat_interval: 일정 압력 구간 index 범위 (기본값 예시: (7000, 43000))
        transform: 데이터 변환 함수
        """
        self.data_dir = data_dir
        self.channel = channel
        self.flat_interval = flat_interval
        self.transform = transform

        # 파일 리스트
        self.file_list = []
        for f in os.listdir(data_dir):
            if f.endswith('.txt'):
                file_path = os.path.join(data_dir, f)
                try:
                    df = pd.read_csv(file_path, nrows=self.flat_interval[1] + 1)  # nrows로 속도 최적화
                    if len(df) >= self.flat_interval[1]:
                        self.file_list.append(f)
                except Exception as e:
                    print(f"[Warning] Skipping {f} due to read error: {e}")
                    continue
        #print(self.file_list)
        # Label 불러오기
        label_df = pd.read_excel(label_excel_path,header=[0, 1, 2])
        #print(label_df.columns.tolist())
        han_yeol_column = label_df[('한의사의 변증 결과 (label 1)', '한의사의 한열 변증 (forced choice)', '한(1)/열(2)')] - 1
        huh_sil_column = label_df[('한의사의 변증 결과 (label 1)', '한의사의 허실 변증 (forced choice)', '허(1)/실(2)')] - 1
        
        han_survey_column = label_df[('설문지의 변증 결과 (label 2)', '한열설문지', '한증_총점')]
        yeol_survey_column = label_df[('설문지의 변증 결과 (label 2)', '한열설문지', '열증_총점')]
        huh_survey_column = label_df[('설문지의 변증 결과 (label 2)', '허실설문지', '허증_총점')]
        sil_survey_column = label_df[('설문지의 변증 결과 (label 2)', '허실설문지', '실증 총점')]
        
        label1 = zip(han_yeol_column, huh_sil_column)
        label2 = zip(han_survey_column, yeol_survey_column, huh_survey_column, sil_survey_column)
        self.labels_dict = dict(zip(label_df[('ID', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2')], zip(label1, label2)))
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # 데이터 읽기
        data = pd.read_csv(file_path)
        #print(file_path)
        #print(data)
        # flat 구간 슬라이싱
        start_idx, end_idx = self.flat_interval
        
        if self.channel == 1:
            pulse_data = data['Ch3_Raw'].values[start_idx:end_idx]
        else:
            pulse_data = []
            for ch in range(1, self.channel + 1):
                pulse_data.append(data[f'Ch{ch}_Raw'].values[start_idx:end_idx])
            pulse_data = np.array(pulse_data)
        #print(pulse_data.shape)
        
        # 데이터 형태 변환 (예: [batch, channel, length] 형태로 변환)
        pulse_data = pulse_data.astype(np.float32)
        
        # 채널별로 평균, 표준편차를 따로 계산하여 정규화
        pulse_data_mean = pulse_data.mean(axis=1, keepdims=True)
        pulse_data_std = pulse_data.std(axis=1, keepdims=True) + 1e-8  # 0으로 나누기 방지

        pulse_data = (pulse_data - pulse_data_mean) / pulse_data_std
        
        
        if self.channel == 1:
            pulse_data = np.expand_dims(pulse_data, axis=0)  # single channel 데이터일 경우
        
        if self.transform:
            pulse_data = self.transform(pulse_data)
            pulse_data = pulse_data.squeeze(0)
        
        # extract subject ID from filename, e.g. 'DMP-RAW_S-001-...'
        filename_no_ext = os.path.splitext(file_name)[0]
        match = re.search(r'_(S-\d+)', filename_no_ext)
        if match:
            subject_id = match.group(1)
        else:
            raise ValueError(f"Invalid file name: {file_name}")
        
        
        label = self.labels_dict.get(subject_id)
        if label is None:
            raise ValueError(f"[Error] subject_id '{subject_id}' not found in labels_dict")
        
        label1_clean = [0 if pd.isna(x) else int(x) for x in label[0]]
                
        label1 = torch.tensor(label1_clean, dtype=torch.long) 
        label2 = torch.tensor(label[1], dtype=torch.float32)
        
        # 회귀 label 정규화 예시
        label2 = (label2 - label2.mean()) / (label2.std() + 1e-8)  # 표준화
        #pulse_data = (pulse_data - np.mean(pulse_data)) / np.std(pulse_data)
        return pulse_data, label1, label2


if __name__ == "__main__":
    # 사용 예시
    data_dir = '/home/gpuadmin/papers/beomseok/Broadcast_and_Media/augmentation/pulse_data/pulse_raw'  # txt 파일이 위치한 디렉토리
    label_excel_path = '/home/gpuadmin/papers/beomseok/Broadcast_and_Media/augmentation/pulse_data/label.xlsx'

    pulse_dataset = PulseDataset(data_dir, label_excel_path)
    loader = DataLoader(pulse_dataset, batch_size=2, shuffle=True)

    for inputs, labels1, labels2 in loader:
        print(inputs.shape, labels1.shape, labels2.shape)
        # 모델 입력으로 활용
        break