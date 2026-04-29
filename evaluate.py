import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import SinhalaMemeModel

# import the test_loader from the dataset loader module
from dataset_loader import test_loader 

def evaluate_model():
    print("Loading model for evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the trained model weights
    model = SinhalaMemeModel().to(device)
    model.load_state_dict(torch.load('sinhala_meme_model.pth', map_location=device))
    model.eval() # set the model to evaluation mode

    all_predictions = []
    all_true_labels = []

    print("Evaluating on unseen test data... Please wait.")
    
    with torch.no_grad():
        for images, input_ids, attention_masks, labels in test_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            # get model predictions
            outputs = model(images, input_ids, attention_masks)
            predictions = torch.argmax(outputs, dim=1)

            # collect predictions and true labels for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 1. generate classification report (f1-score, precision, recall)
    print("\n" + "="*50)
    print("FINAL CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_true_labels, all_predictions, target_names=['Non-Hate (0)', 'Hate (1)']))

    # 2. generate and save the confusion matrix as an image
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hate', 'Hate'], 
                yticklabels=['Non-Hate', 'Hate'])
    plt.xlabel('Predicted by AI')
    plt.ylabel('Actual Truth')
    plt.title('Confusion Matrix - Sinhala Meme Hate Speech')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    print("Confusion matrix graph saved successfully as 'confusion_matrix.png'")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_model()