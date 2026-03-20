# train.py
import torch

import sys
sys.path.append("..")

from ML.config import *
from ML.dataset import get_dataloaders
from ML.model import get_model
from ML.trainer import Trainer

def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(NUM_EPOCHS)

if __name__ == "__main__":
    main()