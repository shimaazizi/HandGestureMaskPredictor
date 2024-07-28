import torch
import numpy as np
import matplotlib.pyplot as plt

def accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    return correct / (target.size(0) * target.size(1) * target.size(2))

def dice_score(pred, target, epsilon=1e-6):
    pred = torch.argmax(pred, dim=1)  
    pred = (pred == 1).float()  
    target = (target == 1).float()  

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()



def visualize_prediction(model, test_loader, device, num_classes=4):
    # Set model to evaluation mode
    model.eval()

    # Get a single batch from the test loader
    images, true_masks = next(iter(test_loader))

    # Move data to the appropriate device
    images = images.to(device)
    true_masks = true_masks.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(images)

    # Convert outputs to class predictions
    _, preds = torch.max(outputs, 1)

    # Move tensors to CPU and convert to numpy arrays
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    preds = preds.cpu().numpy()

    # Plot results for the first image in the batch
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax1.imshow(np.transpose(images[1], (1, 2, 0)))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot true mask
    ax2.imshow(true_masks[1], cmap='jet', vmin=0, vmax=num_classes-1)
    ax2.set_title('True Mask')
    ax2.axis('off')

    # Plot predicted mask
    ax3.imshow(preds[1], cmap='jet', vmin=0, vmax=num_classes-1)
    ax3.set_title('Predicted Mask')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Print unique values in true and predicted masks
    print("Unique values in true mask:", np.unique(true_masks[0]))
    print("Unique values in predicted mask:", np.unique(preds[0]))

