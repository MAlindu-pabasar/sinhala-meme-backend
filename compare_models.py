import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

def test_baseline_models():
    print("Loading Dataset for Baseline Models...")
    # Load dataset 
    df = pd.read_excel('meme_dataset_tesseract.xlsx')

    # Get column names dynamically (0=Image, 1=Label, 2=Text) just like in dataset_loader.py
    label_col = df.columns[1] 
    text_col = df.columns[2]

    # Handle missing values
    df[text_col] = df[text_col].fillna('')

    # Convert labels to binary (Hateful = 1, Non-Hateful = 0)
    df['Label_Binary'] = df[label_col].apply(lambda x: 1 if str(x).strip().lower() == 'hateful' else 0)

    X = df[text_col]
    y = df['Label_Binary']

    # Split data (Test Size 15% - matching your Deep Learning Model split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Convert text to numerical features using TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize the 3 baseline models
    models = {
        "Logistic Regression (Text Only)": LogisticRegression(max_iter=1000),
        "Random Forest       (Text Only)": RandomForestClassifier(random_state=42),
        "Support Vector Mach (Text Only)": SVC(kernel='linear', random_state=42)
    }

    print("\n" + "="*50)
    print("BASELINE MODELS COMPARISON RESULTS")
    print("="*50)
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_vec, y_train)
        
        # Make predictions on test data
        y_pred = model.predict(X_test_vec)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Model: {name}")
        print(f"Accuracy : {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1-Score: {f1*100:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    test_baseline_models()