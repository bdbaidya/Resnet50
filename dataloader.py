import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Cholec80Dataset(Dataset):
    def __init__(self, annotation_dir, video_root, seq_length=16, mode='train'):
        self.samples = []
        self.seq_length = seq_length
        self.transform = self._get_transforms(mode)
        
        # Load all annotation files
        for ann_file in os.listdir(annotation_dir):
            if not ann_file.endswith('.csv'):
                continue
                
            video_name = os.path.splitext(ann_file)[0]
            video_path = os.path.join(video_root, video_name)
            
            with open(os.path.join(annotation_dir, ann_file)) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    start_frame, end_frame, phase = row
                    self.samples.append((
                        video_path,
                        int(start_frame),
                        int(end_frame),
                        int(phase)
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