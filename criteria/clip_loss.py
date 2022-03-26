
import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, image, text):
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity