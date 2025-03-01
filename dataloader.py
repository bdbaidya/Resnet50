import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import PHASE_MAPPING

class Cholec80Dataset(Dataset):
    def __init__(self, annotation_dir, video_root, seq_length=16, stride=8, mode='train'):
        self.samples = []
        self.seq_length = seq_length
        self.stride = stride
        self.transform = self._get_transforms(mode)
        
        # Process each annotation file
        for ann_file in os.listdir(annotation_dir):
            if not ann_file.endswith('.csv'):
                continue
                
            video_name = os.path.splitext(ann_file)[0]
            video_path = os.path.join(video_root, video_name)
            
            # Read annotation file
            with open(os.path.join(annotation_dir, ann_file), 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # Skip header
                
                frames = []
                phases = []
                for row in reader:
                    frame_num = int(row[0])
                    phase = row[1].strip()
                    frames.append(frame_num)
                    phases.append(PHASE_MAPPING[phase])
                
                # Generate sequences
                for start in range(0, len(frames)-seq_length+1, stride):
                    end = start + seq_length
                    sequence_frames = frames[start:end]
                    
                    # Verify consecutive frames
                    if all(sequence_frames[i+1] == sequence_frames[i]+1 
                           for i in range(len(sequence_frames)-1)):
                        label = phases[end-1]
                        self.samples.append( (video_path, sequence_frames[0], label) )

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame, label = self.samples[idx]
        clip = []
        
        for i in range(self.seq_length):
            frame_path = os.path.join(
                video_path, 
                f"frame_{start_frame+i:06d}.jpg"  # Assuming 6-digit zero padding
            )
            img = Image.open(frame_path).convert('RGB')
            clip.append(self.transform(img))
            
        return torch.stack(clip), torch.tensor(label)