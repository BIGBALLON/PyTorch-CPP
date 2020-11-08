import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)

# Don't forget change model to eval mode
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet18.pt")
