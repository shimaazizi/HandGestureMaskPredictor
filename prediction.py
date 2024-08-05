import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.model import UNet


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device('cpu')

# Load image
def load_image(path_image):
  image = Image.open(path_image)
  image = transform(image) # 3, 128, 128 
  image = image.unsqueeze(0) # 1, 3, 128, 128
  img_tensor = image.to(device)
  return img_tensor

# prediction
def predict(img_tensor):
  model = UNet(num_classes = 4).to(device)
  model.load_state_dict(torch.load("model.pth", map_location=device))
  model.eval()
  with torch.no_grad():
    output = model(img_tensor) #1, 4, 128, 128
  _, preds = torch.max(output, dim=1)
  preds = preds.cpu().numpy().T
  preds = plt.imshow(preds)
  plt.show()
  return preds


path_image = "image.jpg"
img_tensor = load_image(path_image)
predict(img_tensor)
