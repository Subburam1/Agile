#!/usr/bin/env python3
"""
Field Detection Model for OCR Text Classification
Trains on field categories to automatically identify different types of fields in documents.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)

class FieldDetectionModel:
    """
    Advanced field detection model for OCR text classification.
    Identifies field categories from extracted text.
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the field detection model."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Model options
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
        
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        
        # Field categories and patterns
        self.field_categories = {
            'personal_info': {
                'patterns': [
                    r'\b(name|full\s*name|first\s*name|last\s*name|candidate\s*name|student\s*name|applicant\s*name|bearer\s*name|holder\s*name)\b',
                    r'\b(address|street|city|state|zip|postal|home\s*address|residential\s*address|permanent\s*address|house\s*no|flat\s*no|building|locality|pin\s*code|country)\b',
                    r'\b(phone|mobile|cell|telephone|contact\s*no|tel|call|dial)\b',
                    r'\b(email|e-mail|mail|email\s*id|email\s*address)\b',
                    r'\b(date\s*of\s*birth|dob|d\.o\.b|birth\s*date|born\s*on|born|age)\b',
                    r'\+?\d{1,4}[\s\-]?\(?\d{3,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4,6}',  # Phone patterns
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date patterns
                    r'\d{1,5}\s+[A-Za-z\s,.-]+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr)\b'  # Address patterns
                ],
                'keywords': ['name', 'address', 'phone', 'email', 'birth', 'contact', 'mobile', 'city', 'state', 'zip', 'street', 'house', 'flat']
            },
            'identification': {
                'patterns': [
                    r'\b(id|identification|passport|license|ssn|aadhar|pan|uid|driving\s*license|dl)\b',
                    r'\b(card\s*number|id\s*number|license\s*number|aadhar\s*number|pan\s*number|passport\s*number)\b',
                    r'\b(social\s*security|driving\s*license)\b',
                    r'\b\d{4}\s*\d{4}\s*\d{4}\b',  # Aadhar pattern
                    r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN pattern
                    r'\b[A-Z]\d{7}\b',  # Passport pattern
                    r'\b[A-Z]{2}\d{2}\s*\d{11}\b'  # Driving License pattern
                ],
                'keywords': ['id', 'passport', 'license', 'ssn', 'identification', 'aadhar', 'pan', 'uid', 'driving', 'card']
            },
            'financial': {
                'patterns': [
                    r'\b(amount|total|sum|cost|price|fee|grand\s*total|net\s*amount)\b',
                    r'\b(account|bank|credit|debit)\b',
                    r'\b(tax|income|salary|wage)\b',
                    r'\b(\$|dollar|usd|currency|₹|rupees?|inr)\b',
                    r'\b(invoice\s*(?:no|number|#)|bill\s*(?:no|number|#)|receipt\s*(?:no|number|#)|inv\s*#?|rcpt\s*#?)\b',
                    r'₹\s*[0-9,]+\.?[0-9]*',  # Indian Rupee amounts
                    r'\$\s*[0-9,]+\.?[0-9]*',  # Dollar amounts
                    r'#\s*[A-Z0-9\-/]{3,}'  # Invoice number pattern
                ],
                'keywords': ['amount', 'total', 'account', 'bank', 'tax', 'money', 'invoice', 'bill', 'receipt', 'rupees', 'cost', 'price']
            },
            'dates_times': {
                'patterns': [
                    r'\b(date|time|day|month|year)\b',
                    r'\b(expiry|expiration|valid|issue)\b',
                    r'\b(from|to|until|during)\b'
                ],
                'keywords': ['date', 'time', 'expiry', 'valid', 'issue']
            },
            'academic': {
                'patterns': [
                    r'\b(grade|score|marks|percentage)\b',
                    r'\b(course|subject|class|degree)\b',
                    r'\b(university|college|school|institute)\b',
                    r'\b(exam|test|assessment|certificate)\b'
                ],
                'keywords': ['grade', 'course', 'university', 'exam', 'certificate']
            },
            'medical': {
                'patterns': [
                    r'\b(diagnosis|treatment|medication|prescription)\b',
                    r'\b(doctor|physician|hospital|clinic)\b',
                    r'\b(blood|pressure|weight|height)\b'
                ],
                'keywords': ['diagnosis', 'doctor', 'hospital', 'blood', 'medical']
            },
            'visual_elements': {
                'patterns': [
                    r'\b(signature|sign\s*here|authorized\s*signature|candidate\s*signature|student\s*signature|applicant\s*signature)\b',
                    r'\b(photo|photograph|picture|passport\s*(?:size\s*)?photo|affix\s*photo|paste\s*photo)\b',
                    r'\b(thumb\s*impression|fingerprint|left\s*thumb|right\s*thumb)\b',
                    r'\b(stamp|seal|official\s*stamp)\b'
                ],
                'keywords': ['signature', 'sign', 'photo', 'photograph', 'picture', 'thumb', 'fingerprint', 'stamp', 'seal']
            },
            'document_metadata': {
                'patterns': [
                    r'\b(document\s*(?:id|number)|reference\s*(?:no|number)|serial\s*(?:no|number)|application\s*(?:no|number)|registration\s*(?:no|number))\b',
                    r'\b(issued\s*(?:date|on)|valid\s*(?:from|until|till)|expiry\s*date|expires\s*on)\b',
                    r'\b(issued\s*by|issued\s*at|place\s*of\s*issue)\b'
                ],
                'keywords': ['document', 'reference', 'serial', 'application', 'registration', 'issued', 'valid', 'expiry', 'place']
            }
        }
        
        logger.info("✅ FieldDetectionModel initialized")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for feature extraction."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, text: str, context: str = "") -> Dict[str, Any]:
        """Extract features from text for field classification."""
        features = {}
        
        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Pattern matching features
        for category, info in self.field_categories.items():
            pattern_matches = 0
            keyword_matches = 0
            
            # Check patterns
            for pattern in info['patterns']:
                if re.search(pattern, text.lower()):
                    pattern_matches += 1
            
            # Check keywords
            text_words = set(text.lower().split())
            keyword_matches = len(text_words.intersection(set(info['keywords'])))
            
            features[f'{category}_patterns'] = pattern_matches
            features[f'{category}_keywords'] = keyword_matches
        
        # Numeric content detection
        features['has_numbers'] = int(bool(re.search(r'\d', text)))
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # Special character detection
        features['has_special_chars'] = int(bool(re.search(r'[^\w\s]', text)))
        features['has_currency'] = int(bool(re.search(r'[\$£€¥]', text)))
        features['has_percentage'] = int(bool(re.search(r'%', text)))
        
        return features
    
    def generate_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate training data from existing suggestions and patterns."""
        logger.info("Generating training data from existing patterns...")
        
        training_samples = []
        
        # Load existing suggestion files if available
        suggestion_files = [
            'suggestions_aadhar.json',
            'suggestions_college_id.json', 
            'suggestions_exam_receipt.json'
        ]
        
        for filename in suggestion_files:
            filepath = Path(filename)
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'field_name' in item:
                                    training_samples.append({
                                        'text': item.get('field_name', ''),
                                        'category': self._classify_field_name(item.get('field_name', '')),
                                        'context': filename.replace('suggestions_', '').replace('.json', '')
                                    })
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_data()
        training_samples.extend(synthetic_data)
        
        if not training_samples:
            logger.warning("No training data available, generating minimal synthetic data")
            training_samples = self._generate_minimal_training_data()
        
        # Convert to DataFrame and ensure unique columns
        df = pd.DataFrame(training_samples)
        
        # Check for duplicate columns and handle them
        if df.columns.duplicated().any():
            logger.warning("Found duplicate columns in training data, deduplicating...")
            df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"Generated {len(df)} training samples with {df['category'].nunique()} categories")
        
        # Ensure required columns exist
        required_cols = ['text', 'context', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Training data missing required columns: {missing_cols}")
        
        return df[['text', 'context']], df['category']
    
    def _classify_field_name(self, field_name: str) -> str:
        """Classify a field name into a category based on patterns."""
        field_lower = field_name.lower()
        
        for category, info in self.field_categories.items():
            for pattern in info['patterns']:
                if re.search(pattern, field_lower):
                    return category
            
            # Check keywords
            for keyword in info['keywords']:
                if keyword in field_lower:
                    return category
        
        return 'other'
    
    def _generate_synthetic_data(self) -> List[Dict[str, str]]:
        """Generate synthetic training data for field detection."""
        synthetic_data = []
        
        # Personal information fields
        personal_fields = [
            "Full Name", "First Name", "Last Name", "Name of Applicant",
            "Address", "Street Address", "City", "State", "Postal Code",
            "Phone Number", "Mobile Number", "Contact Number",
            "Email Address", "Email ID", "Date of Birth", "DOB"
        ]
        
        # Identification fields
        id_fields = [
            "ID Number", "Passport Number", "License Number", "SSN",
            "Card Number", "Registration Number", "Student ID",
            "Employee ID", "Account Number"
        ]
        
        # Financial fields
        financial_fields = [
            "Amount", "Total Amount", "Fee Amount", "Cost", "Price",
            "Account Balance", "Tax Amount", "Income", "Salary"
        ]
        
        # Date/time fields
        date_fields = [
            "Issue Date", "Expiry Date", "Valid Until", "From Date",
            "To Date", "Transaction Date", "Created On"
        ]
        
        # Academic fields
        academic_fields = [
            "Grade", "Score", "Percentage", "Marks Obtained",
            "Course Name", "Subject", "University Name", "Degree",
            "Certificate Number", "Roll Number"
        ]
        
        # Add samples for each category
        for field in personal_fields:
            synthetic_data.append({
                'text': field,
                'category': 'personal_info',
                'context': 'synthetic'
            })
        
        for field in id_fields:
            synthetic_data.append({
                'text': field,
                'category': 'identification',
                'context': 'synthetic'
            })
        
        for field in financial_fields:
            synthetic_data.append({
                'text': field,
                'category': 'financial',
                'context': 'synthetic'
            })
        
        for field in date_fields:
            synthetic_data.append({
                'text': field,
                'category': 'dates_times',
                'context': 'synthetic'
            })
        
        for field in academic_fields:
            synthetic_data.append({
                'text': field,
                'category': 'academic',
                'context': 'synthetic'
            })
        
        return synthetic_data
    
    def _generate_minimal_training_data(self) -> List[Dict[str, str]]:
        """Generate minimal training data as fallback."""
        return [
            {'text': 'Name', 'category': 'personal_info', 'context': 'minimal'},
            {'text': 'Address', 'category': 'personal_info', 'context': 'minimal'},
            {'text': 'ID Number', 'category': 'identification', 'context': 'minimal'},
            {'text': 'Amount', 'category': 'financial', 'context': 'minimal'},
            {'text': 'Date', 'category': 'dates_times', 'context': 'minimal'},
            {'text': 'Grade', 'category': 'academic', 'context': 'minimal'},
        ]
    
    def train(self, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Train the field detection model."""
        logger.info(f"Training field detection model using {model_name}...")
        
        # Generate training data
        X, y = self.generate_training_data()
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # Preprocess text
        X_processed = X['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Create and train pipeline
        if model_name not in self.models:
            model_name = 'random_forest'
            logger.warning(f"Model not found, using default: {model_name}")
        
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.models[model_name])
        ])
        
        pipeline.fit(X_train, y_train_encoded)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Store best model
        self.best_model = pipeline
        self.best_model_name = model_name
        self.is_trained = True
        
        # Generate report
        report = classification_report(
            y_test_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        training_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'num_samples': len(X),
            'num_categories': len(self.label_encoder.classes_),
            'categories': list(self.label_encoder.classes_),
            'classification_report': report,
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Model trained successfully with accuracy: {accuracy:.3f}")
        
        # Save model
        self.save_model()
        
        return training_results
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """Predict field category for given text."""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Predict
        prediction_encoded = self.best_model.predict([processed_text])[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        result = {
            'predicted_category': prediction,
            'original_text': text,
            'processed_text': processed_text
        }
        
        if return_probabilities:
            probabilities = self.best_model.predict_proba([processed_text])[0]
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                category = self.label_encoder.inverse_transform([i])[0]
                prob_dict[category] = float(prob)
            result['probabilities'] = prob_dict
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict categories for multiple texts."""
        return [self.predict(text, return_probabilities=True) for text in texts]
    
    def save_model(self, filename: str = None) -> str:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            filename = f"field_detection_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.best_model,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name,
            'field_categories': self.field_categories,
            'training_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.best_model_name = model_data.get('model_name', 'unknown')
        self.field_categories = model_data.get('field_categories', self.field_categories)
        self.is_trained = True
        
        logger.info(f"✅ Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_name': self.best_model_name,
            'categories': list(self.label_encoder.classes_),
            'num_categories': len(self.label_encoder.classes_),
            'is_trained': self.is_trained
        }


def main():
    """Main function to demonstrate field detection model."""
    print("🧠 Field Detection Model Training Demo")
    print("=" * 50)
    
    # Initialize model
    model = FieldDetectionModel()
    
    # Train model
    print("🔄 Training model...")
    results = model.train('random_forest')
    
    print(f"✅ Training completed!")
    print(f"   Accuracy: {results['accuracy']:.3f}")
    print(f"   Samples: {results['num_samples']}")
    print(f"   Categories: {results['num_categories']}")
    print(f"   Categories: {', '.join(results['categories'])}")
    
    # Test predictions
    test_texts = [
        "Full Name",
        "Phone Number", 
        "Passport Number",
        "Total Amount",
        "Issue Date",
        "Grade Obtained",
        "Blood Pressure"
    ]
    
    print(f"\n🧪 Testing predictions:")
    print("-" * 30)
    
    for text in test_texts:
        prediction = model.predict(text, return_probabilities=True)
        print(f"'{text}' → {prediction['predicted_category']} "
              f"(confidence: {prediction.get('confidence', 0):.3f})")
    
    print(f"\n🎉 Field detection model demo completed!")


if __name__ == "__main__":
    main()