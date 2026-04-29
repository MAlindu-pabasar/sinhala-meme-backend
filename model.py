import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class SinhalaMemeModel(nn.Module):
    def __init__(self):
        super(SinhalaMemeModel, self).__init__()
        
        # 1. Image Model (ResNet50) - .pth ෆයිල් එකේ තියෙන විදිහටම 'vision_branch' ලෙස වෙනස් කළා
        resnet = models.resnet50(weights='DEFAULT')
        # අවසාන layer එක අයින් කරලා ඉතුරු ටික ගන්නවා (කලින් Train කරපු විදිහටම)
        modules = list(resnet.children())[:-1]
        self.vision_branch = nn.Sequential(*modules)
        
        # 2. Text Model (Multilingual BERT) - 'text_branch' ලෙස වෙනස් කළා
        self.text_branch = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # 3. Classifier Head (Combining Image + Text)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract features from the image
        img_out = self.vision_branch(image)
        # පින්තූරයේ දත්ත ටික තනි පේළියකට (Flatten) කරනවා
        img_out = img_out.view(img_out.size(0), -1)
        
        # Extract features from the text
        text_out = self.text_branch(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        # Combine image and text features
        combined_features = torch.cat((img_out, text_out), dim=1)
        
        # Pass the combined features through the classifier
        output = self.classifier(combined_features)
        
        return output