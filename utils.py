import torch
import wandb

def setup_distributed():
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()

def save_checkpoint(model, path, epoch, accuracy):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'val_acc': accuracy
    }, path)
    wandb.save(path)

def validate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return correct / total