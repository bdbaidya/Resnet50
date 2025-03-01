import torch
from model import TemporalResNet
from dataloader import Cholec80Dataset
import wandb
from wandb_config import WANDB_CONFIG

def finetune(args):
    # Initialize WandB
    wandb.login(key=WANDB_CONFIG["API_KEY"])
    wandb.init(
        project=WANDB_CONFIG["PROJECT"],
        entity=WANDB_CONFIG["ENTITY"],
        config=vars(args),
        tags=["fine-tuning"]
    )

    # Load pretrained model
    model = TemporalResNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.pretrained_path))
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # Modify classifier
    model.classifier = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, args.num_classes))
    
    # Dataset with different augmentations
    dataset = Cholec80Dataset(
        args.annotation_path,
        args.video_root,
        seq_length=8,  # Shorter sequences
        mode='finetune'
    )
    
    # Training with smaller batch size
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Optimizer only updates classifier
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(args.epochs):
        # ... similar training logic with validation ...
        wandb.log({"ft_loss": loss, "ft_acc": acc})
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    finetune(args)