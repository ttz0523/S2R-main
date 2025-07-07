import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Img_Mes_Dataset(Dataset):

    def __init__(self, Ien_path, M_txt_path, H=256, W=256):
        super(Img_Mes_Dataset, self).__init__()
        self.H = H
        self.W = W
        self.Ien_path = Ien_path

        self.img_mes_path = M_txt_path

        self.img_label = self.load_annotations(self.img_mes_path)

        self.img = [os.path.join(self.Ien_path, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]

        self.transform = transforms.Compose([
            transforms.Resize((int(self.H), int(self.W))),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def transform_image(self, image):

        image = self.transform(image)

        return image

    def load_annotations(self, ann_file):
        data_infos = {}
        with open(ann_file) as f:
            samples = [x.strip().split('|') for x in f.readlines()]
            for filename, message in samples:
                data_infos[filename] = message
        return data_infos

    def __getitem__(self, index):

        while True:
            image = Image.open(self.img[index]).convert("RGB")
            message = self.label[index]
            image = self.transform_image(image)
            if image is not None:
                return image, message
            index += 1

    def __len__(self):
        return len(self.img)
