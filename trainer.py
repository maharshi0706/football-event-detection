# import torch
# import torch.nn as nn
# from torch.amp import autocast, GradScaler 
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# # from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import accuracy_score, f1_score
# from tqdm import tqdm
# import config

# class Trainer:
#     def __init__(self, model, train_loader, val_loader=None):
#         self.model = model.to(config.DEVICE)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = AdamW(self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
#         self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
#         self.scaler = GradScaler('cuda')  # Fixed: use 'cuda' device type
#         self.writer = None
#         self.best_acc = 0.0
#         self.best_loss = float('inf')  # For early stopping if enabled

#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0.0
#         self.optimizer.zero_grad()

#         for step, (videos, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")):
#             videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

#             print(f"Batch {step+1} - videos shape: {videos.shape}")

#             with autocast(device_type="cuda", dtype=torch.float16):
#                 outputs = self.model(videos)
#                 logits = outputs.logits
#                 # loss = outputs.loss if outputs.loss is not None else nn.CrossEntropyLoss()(logits, labels) / config.ACCUMULATION_STEPS
#                 loss = nn.CrossEntropyLoss()(logits, labels) / config.ACCUMULATION_STEPS
#             self.scaler.scale(loss).backward()

#             if (step + 1) % config.ACCUMULATION_STEPS == 0:
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer.zero_grad()

#             total_loss += loss.item() * config.ACCUMULATION_STEPS

#         avg_loss = total_loss / len(self.train_loader)
#         # self.writer.add_scalar("Loss/train", avg_loss, epoch)
#         print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
#         self.scheduler.step()

#     def validate(self, epoch):
#         if not self.val_loader:
#             print("No validation loader provided.")
#             return 0.0, 0.0, 0.0

#         self.model.eval()
#         preds, trues = [], []
#         total_val_loss = 0.0
#         num_batches = 0

#         with torch.no_grad():
#             for batch_idx, (videos, labels) in enumerate(self.val_loader):
#                 videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

#                 print(f"Val batch {batch_idx+1} - videos shape: {videos.shape}")

#                 # with autocast(device_type="cuda", dtype=torch.float16):  # Fixed: modern autocast
#                 #     logits = self.model(videos)
#                 #     batch_loss = nn.CrossEntropyLoss()(logits, labels)
#                 #     pred = torch.argmax(logits, dim=-1).cpu().numpy()
#                 #     preds.extend(pred)
#                 #     trues.extend(labels.cpu().numpy())

#                 with autocast(device_type="cuda", dtype=torch.float16):
#                     outputs = self.model(videos)
#                     logits = outputs.logits                          
#                     batch_loss = nn.CrossEntropyLoss()(logits, labels)

#                 pred = torch.argmax(logits, dim=-1).cpu().numpy()
#                 preds.extend(pred)
#                 trues.extend(labels.cpu().numpy())
#                 total_val_loss += batch_loss.item()
#                 num_batches += 1

#         avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
#         acc = accuracy_score(trues, preds)
#         f1 = f1_score(trues, preds, average='weighted')

#         # self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
#         # self.writer.add_scalar("Accuracy/val", acc, epoch)
#         # self.writer.add_scalar("F1/val", f1, epoch)
#         print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

#         # Save if best accuracy
#         if acc > self.best_acc:
#             self.best_acc = acc
#             torch.save(self.model.state_dict(), config.CHECKPOINT_DIR / f"best_acc_{acc:.4f}.pth")
#             print("Saved best model (accuracy)")

#         # Save every epoch (overwrite last or keep all — your choice)
#         torch.save(self.model.state_dict(), config.CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")

#         # Optional: Early stopping (uncomment if you want)
#         # if avg_val_loss < self.best_loss:
#         #     self.best_loss = avg_val_loss
#         #     epochs_no_improve = 0
#         # else:
#         #     epochs_no_improve += 1
#         #     if epochs_no_improve >= 5:  # patience=5
#         #         print("Early stopping triggered")
#         #         return True  # signal to stop in fit

#         return acc, f1

#     def fit(self, num_epochs):
#         for epoch in range(num_epochs):
#             self.train_epoch(epoch)
#             self.validate(epoch)
#             # Optional early stopping check (uncomment if added above)
#             # if stop:
#             #     break


import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import glob
import config


class Trainer:
    def __init__(self, model, train_loader, val_loader=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        self.scaler = GradScaler('cuda')
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.start_epoch = 0          

        self._resume_if_checkpoint_exists()

    def _resume_if_checkpoint_exists(self):
        """
        Auto-resume from the latest epoch_N.pth checkpoint if one exists.
        Restores: model weights, optimizer, scheduler, scaler, epoch, best_acc.
        """
        checkpoint_dir = config.CHECKPOINT_DIR
        epoch_checkpoints = sorted(
            glob.glob(str(checkpoint_dir / "epoch_*.pth")),
            key=lambda p: int(p.split("epoch_")[-1].replace(".pth", ""))
        )

        if not epoch_checkpoints:
            print("No checkpoint found. Starting training from scratch.")
            return

        latest = epoch_checkpoints[-1]
        print(f"Resuming from checkpoint: {latest}")

        checkpoint = torch.load(latest, map_location=config.DEVICE)

        # Support both raw state_dict saves and full checkpoint dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.start_epoch = checkpoint["epoch"]          # resume from next epoch
            self.best_acc = checkpoint.get("best_acc", 0.0)
            print(f"Resumed at epoch {self.start_epoch}, best_acc={self.best_acc:.4f}")
        else:
            # Legacy: only model weights were saved (old format)
            self.model.load_state_dict(checkpoint)
            completed_epoch = int(latest.split("epoch_")[-1].replace(".pth", ""))
            self.start_epoch = completed_epoch
            print(f"Loaded model weights only (old format). Resuming from epoch {self.start_epoch}.")
            print("Note: optimizer/scheduler state not restored — LR may differ for first epoch.")

    def _save_checkpoint(self, epoch, acc):
        """Save full checkpoint (model + optimizer + scheduler + scaler + metadata)."""
        checkpoint = {
            "epoch": epoch + 1,                                   
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_acc": self.best_acc,
        }
        path = config.TRAINING_VERSION / f"epoch_{epoch + 1}.pth"
        torch.save(checkpoint, path)
        return path

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (videos, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")):
            videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(videos)
                logits = outputs.logits
                loss = nn.CrossEntropyLoss()(logits, labels) / config.ACCUMULATION_STEPS

            self.scaler.scale(loss).backward()

            if (step + 1) % config.ACCUMULATION_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * config.ACCUMULATION_STEPS

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        self.scheduler.step()

    def validate(self, epoch):
        if not self.val_loader:
            print("No validation loader provided.")
            return 0.0, 0.0

        self.model.eval()
        preds, trues = [], []
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

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
        print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Save best accuracy checkpoint
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(
                self.model.state_dict(),
                config.CHECKPOINT_DIR / f"best_acc_{acc:.4f}.pth"
            )
            print(f"Saved best model (accuracy={acc:.4f})")

        return acc, f1

    def fit(self, num_epochs):
        if self.start_epoch >= num_epochs:
            print(f"Already completed {self.start_epoch}/{num_epochs} epochs. Nothing to do.")
            return

        print(f"Training epochs {self.start_epoch + 1} → {num_epochs}")

        for epoch in range(self.start_epoch, num_epochs):
            self.train_epoch(epoch)
            acc, f1 = self.validate(epoch)

            # Save full checkpoint after every epoch (enables resume)
            saved_path = self._save_checkpoint(epoch, acc)
            print(f"Checkpoint saved: {saved_path}")