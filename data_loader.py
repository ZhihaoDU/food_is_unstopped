import os

from PIL import Image
from torch.utils.data import Dataset


class Food(Dataset):
    def __init__(self, root_dir='../food-101/', mode='train', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        if mode == 'train':
            file_path = os.path.join(root_dir, 'meta/train.txt')
        elif mode == 'eval':
            file_path = os.path.join(root_dir, 'meta/test.txt')
        elif mode == 'test':
            pass
        self.mp_class_name2id = {}
        with open(root_dir + 'meta/classes.txt') as fp:
            for ii, line in enumerate(fp):
                self.mp_class_name2id[line.rstrip()] = ii

        with open(file_path, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]

    def __getitem__(self, index):
        class_name = self.file_list[index].split('/')[0]
        img_path = self.root_dir + 'images/' + self.file_list[index]
        data = Image.open(img_path)
        data = self.transform(data)
        return data, self.mp_class_name2id[class_name]

    def __len__(self):
        return len(self.file_list)
