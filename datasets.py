import os
import glob
import random
import math
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

class ACCORDDataset(object):
    def __init__(self, data_dir, fault_flags, ab_range, time_steps, channels, k, use_normal):
        self.data_dir = data_dir # CWRUデータフォルダ
        self.fault_flags = fault_flags # 故障種別データフラッグ
        self.ab_range = ab_range
        #self.abnormal_flags = abnormal_flags # 異常データフラッグ
        self.time_steps = time_steps # データ長
        self.channels = channels
        self.k = k # k-分割交差検証のサブセット数
        self.val_idx = 0 # 検証サブセット
        self.use_normal = use_normal
        # 正常/異常データの読み取り
        #self.normal_samples, self.abnormal_samples = self._get_samples()
        self.samples = self._get_samples()

    def _get_normal_flags(self, abnormal_flags):
        return [x for x in self.fault_flags if x not in abnormal_flags]

    def _split_files(self, flags):
        kf_samples = {}
        for flag in flags:
            files_dir = os.path.join(self.data_dir, flag, str(self.ab_range[1] - self.ab_range[0]))
            ab_samples = []
            n_samples = []
            for d in tqdm(os.listdir(files_dir), desc='Splitting ACCORD files ({})'.format(flag)):
                f = os.path.join(files_dir, d, 'OUTPUT.ft92')
                arr = np.loadtxt(f, delimiter= ',',skiprows=(1))
                ab_arr = arr[self.ab_range[0]:self.ab_range[1], :self.channels]
                n_arr = arr[(self.ab_range[0] - (self.ab_range[1] - self.ab_range[0])):self.ab_range[0], :self.channels]
                for idx in range(len(ab_arr) - self.time_steps):
                    ab_sample = ab_arr[idx: idx+self.time_steps]
                    ab_samples.append(ab_sample)
                
                if self.use_normal:
                    for idx in range(len(n_arr) - self.time_steps):
                        n_sample = n_arr[idx: idx+self.time_steps]
                        n_samples.append(n_sample)

            random.shuffle(ab_samples)
            k_sample_num = int(math.ceil(len(ab_samples) / float(self.k)))
            if flag not in kf_samples.keys():
                kf_samples[flag] = {'abnormal': [], 'normal': []}
            kf_samples[flag]['abnormal'] = [ab_samples[i:i+k_sample_num] for i in range(0, len(ab_samples), k_sample_num)]
            if self.use_normal:
                random.shuffle(n_samples)
                k_sample_num = int(math.ceil(len(n_samples) / float(self.k)))
                kf_samples[flag]['normal'] = [n_samples[i:i+k_sample_num] for i in range(0, len(n_samples), k_sample_num)]

        return kf_samples

    def _get_samples(self):
        return self._split_files(self.fault_flags)

    def __norm(self, data):
        """正規化
        """

        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    def generate_datasets(self, abnormal_flags, val_idx):
        self.val_idx = val_idx
        
        # 正常データフラッグ(正常なデータと異常データフラッグを除くすべての故障種別データフラッグ)
        normal_flags = self._get_normal_flags(abnormal_flags)

        self.normal_samples_train = None
        self.normal_samples_test = None
        for normal_flag in normal_flags:
            print('Normal flag: {}'.format(normal_flag))
            for idx, s in enumerate(self.samples[normal_flag]['abnormal']):
                if idx != self.val_idx:
                    for sample in tqdm(s, desc='Generating normal training data ({})'.format(idx+1)):
                        if self.normal_samples_train is None:
                            self.normal_samples_train = sample[np.newaxis, :, :]
                        else:
                            self.normal_samples_train = np.vstack((self.normal_samples_train, sample[np.newaxis, :, :]))
                else:
                    for sample in tqdm(s, desc='Generating normal testing data ({})'.format(idx+1)):
                        if self.normal_samples_test is None:
                            self.normal_samples_test = sample[np.newaxis, :, :]
                        else:
                            self.normal_samples_test = np.vstack((self.normal_samples_test, sample[np.newaxis, :, :]))
            
            if self.use_normal:
                for idx, s in enumerate(self.samples[normal_flag]['normal']):
                    if idx != self.val_idx:
                        for sample in tqdm(s, desc='Generating normal training data ({})'.format(idx+1)):
                            if self.normal_samples_train is None:
                                self.normal_samples_train = sample[np.newaxis, :, :]
                            else:
                                self.normal_samples_train = np.vstack((self.normal_samples_train, sample[np.newaxis, :, :]))
                    else:
                        for sample in tqdm(s, desc='Generating normal testing data ({})'.format(idx+1)):
                            if self.normal_samples_test is None:
                                self.normal_samples_test = sample[np.newaxis, :, :]
                            else:
                                self.normal_samples_test = np.vstack((self.normal_samples_test, sample[np.newaxis, :, :]))

        self.normal_samples_train = self.__norm(self.normal_samples_train)
        self.normal_samples_test = self.__norm(self.normal_samples_test)

        self.abnormal_samples_train = None
        self.abnormal_samples_test = None
        for abnormal_flag in abnormal_flags:
            print('Abnormal flag: {}'.format(abnormal_flag))
            for idx, s in enumerate(self.samples[abnormal_flag]['abnormal']):
                if idx != self.val_idx:
                    for sample in tqdm(s, desc='Generating abnormal training data ({})'.format(idx+1)):
                        if self.abnormal_samples_train is None:
                            self.abnormal_samples_train = sample[np.newaxis, :, :]
                        else:
                            self.abnormal_samples_train = np.vstack((self.abnormal_samples_train, sample[np.newaxis, :, :]))
                else:
                    for sample in tqdm(s, desc='Generating abnormal testing data ({})'.format(idx+1)):
                        if self.abnormal_samples_test is None:
                            self.abnormal_samples_test = sample[np.newaxis, :, :]
                        else:
                            self.abnormal_samples_test = np.vstack((self.abnormal_samples_test, sample[np.newaxis, :, :]))
        
        self.abnormal_samples_train = self.__norm(self.abnormal_samples_train)
        self.abnormal_samples_test = self.__norm(self.abnormal_samples_test)

        return {'train': [self.normal_samples_train, self.abnormal_samples_train],
                'test': [self.normal_samples_test, self.abnormal_samples_test]}

class ACCORDDataloader(object):
    def __init__(self, dataset, batch_size):
        self.normal_samples = dataset[0]
        self.abnormal_samples = dataset[1]
        self.batch_size = batch_size
        self.batch_ids = [batch_size, batch_size]

    def __iter__(self): 
        return self
 
    def __next__(self):  
        if self.batch_ids[0] <= len(self.normal_samples):
            if self.batch_ids[1] > len(self.abnormal_samples):
                self.batch_ids[1] = self.batch_size

            neg_samples = self.normal_samples[self.batch_ids[0]-self.batch_size:self.batch_ids[0], :, :]
            pos_samples = self.abnormal_samples[self.batch_ids[1]-self.batch_size:self.batch_ids[1], :, :]

            self.batch_ids[0] += self.batch_size
            self.batch_ids[1] += self.batch_size

            return {'pos_data': pos_samples, 'neg_data': neg_samples}
        else:
            self.batch_ids = [self.batch_size, self.batch_size]
            np.random.shuffle(self.normal_samples)
            np.random.shuffle(self.abnormal_samples)
            raise StopIteration

    def gen_len(self):
        return len(self.normal_samples) // self.batch_size

if __name__ == '__main__':
    data_dir = './data/ACCORD/'
    fault_flags = ['31_GCI', '32_34_GC1P1_3', '35_GC2P']
    abnormal_flags = ['31_GCI']
    ab_range = [1000, 1060]
    time_steps = 30
    channels = 76
    k = 10
    use_normal = True
    val_idx = 2
    dataset = ACCORDDataset(data_dir=data_dir, 
                            fault_flags=fault_flags,
                            ab_range=ab_range,
                            time_steps=time_steps,
                            channels=channels,
                            k=k,
                            use_normal=use_normal)
    datasets = dataset.generate_datasets(abnormal_flags=abnormal_flags, val_idx=val_idx)
    dataloader = ACCORDDataloader(dataset=datasets['train'], batch_size=8)
    for step, data in enumerate(dataloader):
        pos_data, neg_data = data['pos_data'], data['neg_data']
        print('Step: {0}, pos shape: {1}, neg shape: {2}'.format(step, pos_data.shape, neg_data.shape))