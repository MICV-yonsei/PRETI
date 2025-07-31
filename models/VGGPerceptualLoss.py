import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 14], device='cuda'):
        """
        Compute Perceptual Loss using a Pretrained VGG Network
        - layers: list of VGG layer indices to use (e.g., 2: Conv1_2, 7: Conv2_2, 14: Conv3_3, etc.)
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()  # Load pretrained VGG-16 model
        self.selected_layers = layers
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layers) + 1])  # Use only required layers

        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Do not update VGG weights during training

    def forward(self, img1, img2):
        """
        img1: Ground truth image
        img2: Reconstructed image
        """
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            img1 = layer(img1)
            img2 = layer(img2)
            # if i in self.selected_layers:
            #     loss += nn.functional.l1_loss(img1, img2)  # L1 Loss로 Perceptual Loss 계산
            if i in self.selected_layers:
                img1 = F.adaptive_avg_pool2d(img1, (16, 16))  # Reduce feature map size
                img2 = F.adaptive_avg_pool2d(img2, (16, 16))
                loss += nn.functional.l1_loss(img1, img2)
        return loss
