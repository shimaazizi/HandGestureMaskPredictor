import torch
from data.dataset import create_dataloaders
from model.model import UNet
from src.train import train_model, evaluate_model
from utils.utils import visualize_prediction

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def main():
    dataset_path = '/home/shima98/Dataset'
    mask_path = '/home/shima98/New_Mask'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, mask_path)
    
    num_classes = 4
    model = UNet(num_classes).to(device)
    
    trained_model = train_model(model, train_loader, val_loader, test_loader, num_epochs=80, device=device)
    
    test_loss, test_dice, test_acc = evaluate_model(trained_model, test_loader, device)
    print(f'Final Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}, Test Accuracy: {test_acc:.4f}')
    
    visualize_prediction(trained_model, test_loader, device, num_classes=4)
    
    torch.save(trained_model.state_dict(), 'unet_model.pth')

if __name__ == "__main__":
    main()