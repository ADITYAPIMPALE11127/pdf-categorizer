import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

class SVM_Classifier:
    def __init__(self, kernel='rbf', C=1.0, probability=True):
        """
        Initialize SVM Classifier for document categorization
        
        Parameters:
        - kernel: 'rbf' (recommended for text) or 'linear'
        - C: Regularization parameter (higher = stricter classification)
        - probability: Whether to enable probability estimates
        """
        self.classifier = svm.SVC(
            kernel=kernel,
            C=C,
            probability=probability,
            random_state=42
        )
        self.classes_ = None
        self.is_trained = False
        
    def train(self, features, labels, test_size=0.2):
        """
        Train SVM classifier
        
        Parameters:
        - features: PCA-reduced features (n_samples, n_features)
        - labels: Document categories (n_samples,)
        - test_size: Proportion for test split (0.2 = 20%)
        
        Returns: Training accuracy and test accuracy
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.classes_ = self.classifier.classes_
        self.is_trained = True
        
        # Get predictions
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classes': self.classes_.tolist(),
            'n_samples': len(features),
            'support_vectors': len(self.classifier.support_vectors_)
        }
    
    def predict(self, features, return_probabilities=True):
        """
        Predict categories for new documents
        
        Parameters:
        - features: PCA-reduced features of new documents
        - return_probabilities: Whether to return confidence scores
        
        Returns: Predictions and (optionally) probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        predictions = self.classifier.predict(features)
        
        if return_probabilities:
            probabilities = self.classifier.predict_proba(features)
            return predictions, probabilities
        else:
            return predictions
    
    def predict_single(self, features, document_name=""):
        """
        Predict category for a single document with detailed output
        """
        prediction, probabilities = self.predict([features], return_probabilities=True)
        
        # Get confidence scores for all classes
        confidence_scores = {}
        for class_name, prob in zip(self.classes_, probabilities[0]):
            confidence_scores[class_name] = prob
        
        # Get top category
        top_category = prediction[0]
        top_confidence = max(probabilities[0])
        
        return {
            'document': document_name,
            'predicted_category': top_category,
            'confidence': top_confidence,
            'all_scores': confidence_scores,
            'is_confident': top_confidence > 0.7  # 70% threshold
        }
    
    def get_classification_report(self, features, labels):
        """
        Generate detailed classification report
        """
        predictions = self.predict(features, return_probabilities=False)
        report = classification_report(labels, predictions, output_dict=True)
        return report
    
    def get_support_vector_info(self):
        """
        Get information about support vectors
        """
        if not self.is_trained:
            return {}
        
        return {
            'n_support_vectors': len(self.classifier.support_vectors_),
            'n_support_per_class': self.classifier.n_support_.tolist(),
            'classes_with_support': self.classifier.classes_.tolist()
        }
    
    def save_model(self, filepath):
        """
        Save trained model to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, filepath):
        """
        Load trained model from file
        """
        with open(filepath, 'rb') as f:
            self.classifier = pickle.load(f)
        self.classes_ = self.classifier.classes_
        self.is_trained = True

# ==================== TRAINING DATA GENERATOR ====================
def create_training_data():
    """
    Create sample training data for demonstration
    In real scenario, you would load actual labeled documents
    """
    # Simulated training data (you'll replace with actual labeled PDFs)
    # Format: List of (text, category) pairs
    training_docs = [
        # Contracts
        ("agreement terms party obligation clause indemnify", "Contract"),
        ("contract services payment termination confidentiality", "Contract"),
        ("agreement party terms conditions herein", "Contract"),
        
        # Invoices
        ("invoice payment amount due subtotal tax", "Invoice"),
        ("bill amount payment terms net balance", "Invoice"),
        ("invoice total due date payment instructions", "Invoice"),
        
        # Court Documents
        ("court motion plaintiff defendant jurisdiction relief", "Court"),
        ("filing case motion order court hearing", "Court"),
        ("plaintiff defendant motion summary judgment", "Court"),
        
        # Reports
        ("analysis findings recommendation data conclusion", "Report"),
        ("report analysis executive summary metrics", "Report"),
        ("analysis data findings recommendations summary", "Report"),
        
        # Other (catch-all category)
        ("document information details notes comments", "Other"),
        ("file content information data details", "Other"),
        ("notes comments observations remarks", "Other")
    ]
    
    return training_docs

def prepare_training_features(training_docs, tfidf_vectorizer, pca_reducer):
    """
    Prepare training features from labeled documents
    """
    texts = [doc[0] for doc in training_docs]
    labels = [doc[1] for doc in training_docs]
    
    # Apply TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Apply PCA
    pca_features = pca_reducer.fit_transform(tfidf_matrix)
    
    return pca_features, labels, tfidf_vectorizer, pca_reducer