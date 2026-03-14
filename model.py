import timm
from transformers import VideoMAEForVideoClassification
import config
import torch.nn as nn

def get_model():
    model = VideoMAEForVideoClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.config.hidden_size, config.NUM_CLASSES)
    )
    # model = timm.create_model(
    #     'mvitv2_small',
    #     pretrained=True,
    #     num_classes=config.NUM_CLASSES
    # )
    
    # # Optional: freeze early layers
    # for name, param in model.named_parameters():
    #     if "blocks" in name and int(name.split(".")[1]) < 4:
    #         param.requires_grad = False
    
    return model