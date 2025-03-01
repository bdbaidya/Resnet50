import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import TemporalResNet
from dataloader import Cholec80Dataset
import wandb
from wandb_config import WANDB_CONFIG  # Local config

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Initialize W&B on main process
    if rank == 0:
        wandb.login(key=WANDB_CONFIG["API_KEY"])
        wandb.init(
            project=WANDB_CONFIG["PROJECT"],
            entity=WANDB_CONFIG["ENTITY"],
            config=args.__dict__
        )
    
    # Dataset and loader
    dataset = Cholec80Dataset(
        args.annotation_path,
        args.video_root,
        seq_length=args.seq_length,
        mode='train'
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup
    model = TemporalResNet(args.num_classes).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        # Training loop
        for inputs, labels in loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_acc = validate(model, loader, criterion, rank)
        
        # Checkpointing and logging
        if rank == 0:
            wandb.log({"epoch": epoch, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.module.state_dict(), f"{args.save_dir}/best_model.pth")
                wandb.save(f"{args.save_dir}/best_model.pth")
            
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.module.state_dict(), f"{args.save_dir}/epoch_{epoch+1}.pth")
        
    dist.destroy_process_group()

def validate(model, loader, criterion, rank):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = torch.tensor(correct / total).to(rank)
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    return acc.item() / dist.get_world_size()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_interval", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.multiprocessing.spawn(
        train,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )