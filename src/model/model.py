import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet50_VGG16_Fusion(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.resnet.fc = nn.Identity()        
        resnet_out = 2048                    


        self.vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier.children())[:-1]
        )
        vgg_out = 4096                        

       
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vgg.parameters():
            param.requires_grad = False

   
        self.fc = nn.Sequential(
            nn.Linear(resnet_out + vgg_out, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)    
        f2 = self.vgg(x)       
        fused = torch.cat((f1, f2), dim=1)
        return self.fc(fused)



def get_model(name: str = "fusion", num_classes: int = 2, pretrained: bool = True) -> nn.Module:

    if name == "fusion" or name == "resnet50_vgg16":
        return ResNet50_VGG16_Fusion(num_classes=num_classes, pretrained=pretrained)

    elif name == "mobilenet_v2":  
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model architecture: {name}")
