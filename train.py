import os
# Ensure this environment variable is set before importing other libraries to prevent runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer

# Import custom dataset and model modules
from dataset_loader import SinhalaMemeDataset
from model import SinhalaMemeModel

def main():
    # 1. Set up device agnostic code (CUDA/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load the tokenizer (Text to token IDs)
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # 3. Prepare dataset and transformations
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Load dataset from Excel and image directory
    full_dataset = SinhalaMemeDataset(excel_file='meme_dataset_tesseract.xlsx', 
                                      img_dir='processed_dataset', 
                                      transform=transform)

    # Split the dataset (Train: 700, Val: 150, Test: 150)
    train_set, val_set, test_set = random_split(
        full_dataset, [700, 150, 150], generator=torch.Generator().manual_seed(42)
    )

    # Configure DataLoader with a batch size of 8 to fit within VRAM constraints
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    # 4. Initialize the model
    print("Loading the AI model...")
    model = SinhalaMemeModel().to(device)

    # 5. Define Loss function and Optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) 

    # 6. Start the training loop
    epochs = 15
    print("\nStarting training loop...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for i, (images, texts, labels) in enumerate(train_loader):
            # Move batch to device
            images = images.to(device)
            labels = labels.to(device)

            # Tokenize text batch and move to device
            encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, input_ids, attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Calculate batch metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            # Print statistics every 10 steps
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate and print epoch accuracy
        epoch_acc = (correct_predictions.double() / total_samples) * 100
        print(f"Epoch {epoch+1} completed | Average Loss: {total_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%\n")

    # 7. Save the trained model weights
    print("Training finished successfully!")
    torch.save(model.state_dict(), 'sinhala_meme_model.pth')
    print("Model successfully saved as 'sinhala_meme_model.pth'")

if __name__ == "__main__":
    main()