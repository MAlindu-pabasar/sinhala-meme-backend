import torch
from torchvision import transforms
from PIL import Image
import warnings
from transformers import BertTokenizer

# Import the SinhalaMemeModel class from the model module
from model import SinhalaMemeModel

# Suppress PyTorch warnings for a cleaner console output
warnings.filterwarnings("ignore")

def test_single_meme(image_path, meme_text):
    print("\nLoading AI model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the AI model
    model = SinhalaMemeModel().to(device)
    # Set weights_only=True to suppress security warnings during torch.load
    model.load_state_dict(torch.load('sinhala_meme_model.pth', map_location=device))
    model.eval()

    # 2. Prepare the image for the model
    try:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device) 
    except FileNotFoundError:
        print(f"Error: The image '{image_path}' could not be found. Please check the file path.")
        return {"result": "ERROR: File Not Found", "confidence": 0.0}

    # 3. Prepare the text for the model using BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encoding = tokenizer(meme_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 4. Run inference and calculate confidence score
    print("Analyzing meme...\n")
    with torch.no_grad():
        outputs = model(image, input_ids, attention_mask)
        
        # Calculate confidence score using softmax activation
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence_score = torch.max(probabilities).item() * 100 # Convert to percentage
        confidence_score = round(confidence_score, 2) # Round to 2 decimal places
        
        prediction = torch.argmax(outputs, dim=1).item()

    # 5. Display the result and return the payload to the API
    print("="*50)
    if prediction == 1:
        print(f"AI Prediction: HATEFUL (Confidence: {confidence_score}%)")
        print("="*50 + "\n")
        return {"result": "HATEFUL", "confidence": confidence_score} 
    else:
        print(f"AI Prediction: NON-HATEFUL (Confidence: {confidence_score}%)")
        print("="*50 + "\n")
        return {"result": "NON-HATEFUL", "confidence": confidence_score}

# Interactive testing via terminal
if __name__ == "__main__":
    print("-" * 50)
    print("  Sinhala Meme Hate Speech Detector - LIVE TEST  ")
    print("-" * 50)
    
    img_path = input("Enter image path (e.g., test1.jpg) : ")
    text = input("Enter meme text (in Sinhala)         : ")
    
    result_data = test_single_meme(img_path, text)
    print(f"Final Output: {result_data}")