import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Cholec80Dataset(Dataset):
    def __init__(self, annotation_path, video_root, seq_length=16, mode='train'):
        self.samples = []
        self.seq_length = seq_length
        self.transform = self._get_transforms(mode)
        
        with open(annotation_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4: continue  # Skip header
                video_path, start, end, label = row
                self.samples.append((
                    os.path.join(video_root, video_path),
                    int(start),
                    int(end),
                    int(label)
                ))

    def __len__(self):
        return len(self.samples)

    def _get_transforms(self, mode):
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        path, start, end, label = self.samples[idx]
        frames = sorted(os.listdir(path))[start:end+1]
        indices = torch.linspace(0, len(frames)-1, self.seq_length).long()
        return torch.stack([self.transform(Image.open(os.path.join(path, frames[i]))) for i in indices]), label