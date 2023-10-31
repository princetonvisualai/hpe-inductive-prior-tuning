from torchvision import models
import torch

#  I am assuming we have internet access here
# models.vgg16(pretrained=True).features
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "vgg_model_pretrained_16")

#vgg = models.vgg16(pretrained=False)
#vgg.load_state_dict(torch.load("vgg_model"))