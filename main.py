import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import TemporalResNet
from dataloader import Cholec80Dataset
from config import WANDB_CONFIG
import wandb

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Initialize WandB on main process
    if rank == 0:
        wandb.login(key=WANDB_CONFIG["API_KEY"])
        wandb.init(
            project=WANDB_CONFIG["PROJECT"],
            entity=WANDB_CONFIG["ENTITY"],
            config=vars(args)
        )
    
    # Dataset and DataLoader
    dataset = Cholec80Dataset(
        args.annotation_dir,
        args.video_root,
        seq_length=args.seq_length,
        stride=args.stride
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup
    model = TemporalResNet().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation and logging (omitted for brevity)
        
        # Save checkpoints
        if rank == 0 and (epoch+1) % args.save_interval == 0:
            ckpt_path = f"{args.save_dir}/epoch_{epoch+1}.pt"
            torch.save(model.module.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
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