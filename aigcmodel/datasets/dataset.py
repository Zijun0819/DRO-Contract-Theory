import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from .data_augment import PairCompose, PairRandomCrop, PairToTensor
from torchvision.transforms import ToPILImage, ToTensor, Compose


class LLdataset:
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def server_get_loaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def get_evaluation_loaders(self):
        eval_dataset = EvaluationDataset(self.config.data.eval_dir)
        val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return val_loader

    def fl_get_loaders(self):
        train_loader = self.iid_partition_loader(self.train_dataset,
                                                 batch_size=self.config.training.fl_batch_size,
                                                 n_clients=self.config.training.fl_clients)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def iid_partition_loader(self, dataset, batch_size, n_clients):
        """
        partition the dataset into a dataloader for each client, iid style
        """
        m = len(dataset)
        assert m % n_clients == 0
        m_per_client = m // n_clients
        assert m_per_client % batch_size == 0

        client_data = torch.utils.data.random_split(
            dataset,
            [m_per_client for x in range(n_clients)]
        )
        client_loader = [
            torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, pin_memory=True)
            for x in client_data
        ]
        return client_loader


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()

        self.dir = dir
        with open(dir, "r") as file:
            lines = file.readlines()  # 逐行读取文件内容
            data_list = [line.strip() for line in lines]  # 去除换行符并生成列表
        self.input_names = data_list
        self.transforms = Compose([
            ToTensor()
        ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        img_id = re.split("\\\\", input_name)[-1][:-4]
        #  if self.dir else PIL.Image.open(input_name)  if self.dir else PIL.Image.open(gt_name)
        input_img = Image.open(input_name)

        input_img = self.transforms(input_img)
        input_img = input_img[:3, ...]

        return input_img, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
