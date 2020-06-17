import cv2
import numpy as np
import random
import pickle


class BatchGenerator:

    @staticmethod
    def to_one_hot(img, n_cls, onehot):
        one_hot_mask = np.zeros((img.shape[0], img.shape[1], n_cls))
        masks = []
        for color in onehot:
            mask = np.all(img == color, axis=-1)
            masks.append(mask)
            one_hot_mask[mask] = onehot[color]

        final_mask = np.logical_not(sum(masks))
        one_hot_mask[final_mask] = onehot[(1, 1, 1)]

        return one_hot_mask.astype(int)

    def __init__(self, onehot, txt_filepath, size, n_cls, batch_size):
        self.lines = []
        for line in open(txt_filepath, 'r').readlines():
            line = line.strip()
            if len(line) > 0:
                self.lines.append(line)
        self.height = size[0]
        self.width = size[1]
        self.n_cls = n_cls
        self.batch_size = batch_size
        self.onehot = onehot
        self.i = 0

    def get_sample(self):
        if self.i == 0:
            random.shuffle(self.lines)
        orig_filepath, gt_filepath, hed_path = self.lines[self.i].replace(' ', '').split(',')
        # print(hed_path)
        orig = cv2.imread(orig_filepath)  # 1 and 3 channels swapped
        orig = cv2.resize(orig, (self.width, self.height))
        gt = cv2.imread(gt_filepath)
        gt = cv2.resize(gt, (self.width, self.height))
        #hed_img = cv2.imread(hed_path, 0)
        #print(orig_filepath, gt_filepath, hed_path)
        # print(orig.shape)
        # print(gt.shape)
        # print(hed_img.shape)
        # hed = cv2.resize(hed_img, (self.width, self.height)).reshape(self.height, self.width, 1)
        gt = BatchGenerator.to_one_hot(gt, self.n_cls, self.onehot)  # + neutral class
        #print(np.unique(gt.reshape(-1, gt.shape[-1]), axis=0, return_counts=True))
        self.i = (self.i + 1) % len(self.lines)
        return orig / 255, gt

    def get_batch(self):
        while True:
            orig_batch = np.zeros((self.batch_size, self.height, self.width, 3))
            gt_batch = np.zeros((self.batch_size, self.height, self.width, self.n_cls))
            #hed_batch = np.zeros(((self.batch_size, self.height, self.width, 1)))
            for i in range(self.batch_size):
                orig, gt = self.get_sample()
                orig_batch[i] = orig
                gt_batch[i] = gt
                #hed_batch[i] = hed
            #print([orig_batch.shape, hed_batch.shape, gt_batch.shape])
            yield orig_batch, gt_batch

    def get_size(self):
        return len(self.lines)


"""
num_classes = 29
batch_size = 8
size = (128, 224)
with open('one_hot_dict.pickle', 'rb') as handle:
    onehot = pickle.load(handle)

val_gen = BatchGenerator(onehot, 'val.txt', size, num_classes, batch_size)

while True:
    print(val_gen.get_size())
    break
    print('Batch Start')
    for i, x in enumerate(batch):
        print(x[0][0].shape, x[0][1].shape, x[1].shape, i)
    print('Batch End')
"""
