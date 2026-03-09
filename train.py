# train.py
import torch
from config import *
from dataset import get_dataloaders
from model import get_model
from trainer import Trainer

def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(NUM_EPOCHS)

if __name__ == "__main__":
    main()