import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from transformers import BertTokenizer

class SinhalaMemeDataset(Dataset):
    def __init__(self, excel_file, img_dir, tokenizer, transform=None):
        self.data = pd.read_excel(excel_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load the image
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')
        
        # load the text (column 3 in the excel sheet)
        text = str(self.data.iloc[idx, 2])
        
        # convert the label to binary: 1 for 'hateful', 0 otherwise (column 2)
        label_text = str(self.data.iloc[idx, 1]).strip().lower()
        if label_text == 'hateful':
            label = 1
        else:
            label = 0

        # apply image transformations if provided
        if self.transform:
            image = self.transform(image)

        # tokenize the text for BERT
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        return image, encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(label)

# 1. define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 2. initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 3. create the full dataset
full_dataset = SinhalaMemeDataset(
    excel_file='meme_dataset_tesseract.xlsx', 
    img_dir='processed_dataset/',             
    tokenizer=tokenizer, 
    transform=transform
)

# 4. split data based on standard ratio (70% train, 15% validation, 15% test)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size # remaining data

# randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 5. create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# print the sizes to verify the split
print(f"Total Dataset Size: {total_size}")
print(f"Training Set (70%): {len(train_dataset)} memes")
print(f"Validation Set (15%): {len(val_dataset)} memes")
print(f"Testing Set (15%): {len(test_dataset)} memes")