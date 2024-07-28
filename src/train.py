import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import accuracy, dice_score

def train_model(model, train_loader, val_loader, test_loader, num_classes=4, num_epochs=80, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_acc = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            acc = accuracy(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice.item()
            running_acc += acc.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_score(outputs, masks)
                acc = accuracy(outputs, masks)

                val_loss += loss.item()
                val_dice += dice.item()
                val_acc += acc.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}, Validation Accuracy: {val_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'unet_model.pth')

    # Final evaluation on test set
    test_loss, test_dice, test_acc = evaluate_model(model, test_loader, device)
    print(f'Final Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}, Test Accuracy: {test_acc:.4f}')

    return model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_dice = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            acc = accuracy(outputs, masks)

            test_loss += loss.item()
            test_dice += dice.item()
            test_acc += acc.item()

    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    test_acc /= len(test_loader)

    return test_loss, test_dice, test_acc