import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Food(Dataset):
    def __init__(self, root_dir='../food-101/', mode='train', transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(20),
                transforms.RandomRotation(),
                transforms.Resize(size=224),  # Let smaller edge match
                transforms.RandomCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
        else :
            self.transform = transform
        self.root_dir = root_dir
        if mode == 'train':
            file_path = os.path.join(root_dir, 'meta/train.txt')
        elif mode == 'val':
            file_path = os.path.join(root_dir, 'meta/test.txt')
        elif mode == 'test':
            pass
        self.mp_class_name2id = {}
        with open(root_dir + 'meta/classes.txt') as fp:
            for ii, line in enumerate(fp):
                self.mp_class_name2id[line.rstrip()] = ii

        with open(file_path, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]


        #self.file_list = self.file_list[0:8]
    def __getitem__(self, index):
        class_name = self.file_list[index].split('/')[0]
        img_path = self.root_dir + 'images/' + self.file_list[index]+'.jpg'
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transform(data)
        return data, self.mp_class_name2id[class_name]

    def __len__(self):
        return len(self.file_list)


def __main__():
    train_transforms = transforms.Compose([
        transforms.Resize(size=448),  # Let smaller edge match
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
