import io
import torch
from PIL import Image
from torchvision import transforms
from model.model import UNet
import matplotlib.pyplot as plt


device = torch.device('cpu')

def segmentator():
  
  model = UNet(num_classes = 4).to(device)
  model.load_state_dict(torch.load("/home/shima98/HandGestureMaskPredictor/unet_model.pth", map_location=device, weights_only=False))
  model.eval()
  return model 


def get_segments(model, binary_image):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")


    preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image)
    image = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    _, preds = torch.max(output, dim=1)
    preds = preds.cpu().numpy().T
    preds = plt.imshow(preds)
    plt.show()
    return preds


