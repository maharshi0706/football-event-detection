import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import config

class Trainer:
    def __init__(self, model, train_loader, val_loader=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        self.scaler = GradScaler('cuda')  # Fixed: use 'cuda' device type
        self.writer = None
        self.best_acc = 0.0
        self.best_loss = float('inf')  # For early stopping if enabled

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (videos, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")):
            videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

            print(f"Batch {step+1} - videos shape: {videos.shape}")

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(videos)
                logits = outputs.logits
                # loss = outputs.loss if outputs.loss is not None else nn.CrossEntropyLoss()(logits, labels) / config.ACCUMULATION_STEPS
                loss = nn.CrossEntropyLoss()(logits, labels) / config.ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()

            if (step + 1) % config.ACCUMULATION_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * config.ACCUMULATION_STEPS

        avg_loss = total_loss / len(self.train_loader)
        # self.writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        self.scheduler.step()

    def validate(self, epoch):
        if not self.val_loader:
            print("No validation loader provided.")
            return 0.0, 0.0, 0.0

        self.model.eval()
        preds, trues = [], []
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(self.val_loader):
                videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

                print(f"Val batch {batch_idx+1} - videos shape: {videos.shape}")

                # with autocast(device_type="cuda", dtype=torch.float16):  # Fixed: modern autocast
                #     logits = self.model(videos)
                #     batch_loss = nn.CrossEntropyLoss()(logits, labels)
                #     pred = torch.argmax(logits, dim=-1).cpu().numpy()
                #     preds.extend(pred)
                #     trues.extend(labels.cpu().numpy())

                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(videos)
                    logits = outputs.logits                          
                    batch_loss = nn.CrossEntropyLoss()(logits, labels)

                pred = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.extend(pred)
                trues.extend(labels.cpu().numpy())
                total_val_loss += batch_loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='weighted')

        # self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
        # self.writer.add_scalar("Accuracy/val", acc, epoch)
        # self.writer.add_scalar("F1/val", f1, epoch)
        print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Save if best accuracy
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.model.state_dict(), config.CHECKPOINT_DIR / f"best_acc_{acc:.4f}.pth")
            print("Saved best model (accuracy)")

        # Save every epoch (overwrite last or keep all — your choice)
        torch.save(self.model.state_dict(), config.CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")

        # Optional: Early stopping (uncomment if you want)
        # if avg_val_loss < self.best_loss:
        #     self.best_loss = avg_val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= 5:  # patience=5
        #         print("Early stopping triggered")
        #         return True  # signal to stop in fit

        return acc, f1

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            # Optional early stopping check (uncomment if added above)
            # if stop:
            #     break