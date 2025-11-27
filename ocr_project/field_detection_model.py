#!/usr/bin/env python3
"""
Field Detection Model for Training Field Categories from Extracted OCR Text
Advanced ML system for automatic field detection and categorization from OCR text.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing components
try:
    from ocr.rag_field_suggestion import DocumentFieldKnowledgeBase, FieldPattern, FieldSuggestion
    from train_document_classifier import DocumentClassifierTrainer
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")

@dataclass
class FieldCategory:
    """Represents a field category with its properties."""
    category_name: str
    category_type: str  # e.g., 'personal', 'identification', 'financial', 'address'
    field_names: List[str]
    patterns: List[str]
    keywords: List[str]
    examples: List[str]
    description: str
    confidence_threshold: float = 0.7

@dataclass
class FieldDetectionResult:
    """Result of field detection with category information."""
    field_name: str
    field_category: str
    category_type: str
    detected_value: Optional[str]
    confidence: float
    context: str
    pattern_matched: str
    reasoning: str

class FieldCategoryClassifier:
    """Advanced field category classifier for OCR text."""
    
    def __init__(self):
        """Initialize the field category classifier."""
        self.field_categories = self._initialize_field_categories()
        self.category_encoder = LabelEncoder()
        self.field_classifier = None
        self.category_classifier = None
        self.vectorizer = None
        self.trained_models = {}
        
        # Model persistence
        self.models_dir = Path("models/field_detection")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎯 Field Category Classifier initialized")
        print(f"📁 Models directory: {self.models_dir}")
    
    def _initialize_field_categories(self) -> Dict[str, FieldCategory]:
        """Initialize comprehensive field categories."""
        categories = {
            # Personal Information Fields
            'PERSONAL_NAME': FieldCategory(
                category_name='full_name',
                category_type='personal',
                field_names=['name', 'full_name', 'person_name', 'applicant_name', 'holder_name'],
                patterns=[
                    r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                    r'(?:Name|NAME|Full Name|FULL NAME)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'MR\\.?|MS\\.?|MRS\\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
                ],
                keywords=['name', 'full name', 'applicant', 'holder', 'person', 'mr', 'mrs', 'ms'],
                examples=['RAJESH KUMAR', 'Priya Sharma', 'MR. AMIT SINGH'],
                description='Person full name field detection'
            ),
            
            'PERSONAL_FATHER_NAME': FieldCategory(
                category_name='father_name',
                category_type='personal',
                field_names=['father_name', 'father', 's/o', 'son_of'],
                patterns=[
                    r'(?:S/O|s/o|Son of|FATHER|Father)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'Father[\s\'s]*Name[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                ],
                keywords=['s/o', 'son of', 'father', 'father name'],
                examples=['S/O: RAM KUMAR', 'Father: VIJAY SINGH'],
                description='Father name field detection'
            ),
            
            'PERSONAL_DOB': FieldCategory(
                category_name='date_of_birth',
                category_type='personal',
                field_names=['dob', 'date_of_birth', 'birth_date', 'born'],
                patterns=[
                    r'(?:DOB|Date of Birth|Birth Date)[\s:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
                    r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
                    r'(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
                ],
                keywords=['dob', 'date of birth', 'birth date', 'born on'],
                examples=['DOB: 15/08/1990', '15-08-1990', 'Born: 1990-08-15'],
                description='Date of birth field detection'
            ),
            
            # Identification Document Fields
            'ID_AADHAR': FieldCategory(
                category_name='aadhar_number',
                category_type='identification',
                field_names=['aadhar', 'aadhar_number', 'uid', 'unique_id'],
                patterns=[
                    r'(?:Aadhar|AADHAR|UID)[\s:]*(\d{4}\s*\d{4}\s*\d{4})',
                    r'(\d{4}\s*\d{4}\s*\d{4})',
                    r'(?:Enrollment|Aadhaar)\s*(?:No|Number)[\s:]*(\d{4}\s*\d{4}\s*\d{4})'
                ],
                keywords=['aadhar', 'aadhaar', 'uid', 'enrollment', 'unique identification'],
                examples=['1234 5678 9012', 'Aadhar: 1234 5678 9012'],
                description='Aadhar number field detection'
            ),
            
            'ID_PAN': FieldCategory(
                category_name='pan_number',
                category_type='identification',
                field_names=['pan', 'pan_number', 'permanent_account'],
                patterns=[
                    r'(?:PAN|Pan)[\s:]*([A-Z]{5}\d{4}[A-Z])',
                    r'([A-Z]{5}\d{4}[A-Z])',
                    r'Permanent\s*Account\s*Number[\s:]*([A-Z]{5}\d{4}[A-Z])'
                ],
                keywords=['pan', 'permanent account number', 'income tax'],
                examples=['ABCDE1234F', 'PAN: ABCDE1234F'],
                description='PAN card number field detection'
            ),
            
            'ID_VOTER': FieldCategory(
                category_name='voter_id',
                category_type='identification',
                field_names=['voter_id', 'epic', 'electoral_roll'],
                patterns=[
                    r'(?:EPIC|Epic|Voter)[\s:]*([A-Z]{3}\d{7})',
                    r'([A-Z]{3}\d{7})',
                    r'Electoral\s*Roll[\s:]*([A-Z]{3}\d{7})'
                ],
                keywords=['epic', 'voter id', 'electoral', 'election commission'],
                examples=['ABC1234567', 'EPIC: ABC1234567'],
                description='Voter ID field detection'
            ),
            
            # Address Fields
            'ADDRESS_FULL': FieldCategory(
                category_name='full_address',
                category_type='address',
                field_names=['address', 'full_address', 'residential_address'],
                patterns=[
                    r'(?:Address|ADDRESS)[\s:]+(.+?)(?:\n|$)',
                    r'(?:Residential|Permanent)\s*Address[\s:]+(.+?)(?:\n|$)'
                ],
                keywords=['address', 'residential', 'permanent', 'location'],
                examples=['123 MG Road, Bangalore 560001', 'Address: 456 Park Street'],
                description='Full address field detection'
            ),
            
            'ADDRESS_PIN': FieldCategory(
                category_name='pin_code',
                category_type='address',
                field_names=['pin', 'pincode', 'postal_code', 'zip'],
                patterns=[
                    r'(?:PIN|Pin|Pincode)[\s:]*(\d{6})',
                    r'(\d{6})',
                    r'Postal\s*Code[\s:]*(\d{6})'
                ],
                keywords=['pin', 'pincode', 'postal code', 'zip'],
                examples=['560001', 'PIN: 560001', 'Pincode: 400001'],
                description='PIN code field detection'
            ),
            
            # Contact Information Fields
            'CONTACT_MOBILE': FieldCategory(
                category_name='mobile_number',
                category_type='contact',
                field_names=['mobile', 'phone', 'contact_number'],
                patterns=[
                    r'(?:Mobile|Phone|Contact)[\s:]*(\+91[\s-]?\d{10})',
                    r'(\+91[\s-]?\d{10})',
                    r'(\d{10})',
                    r'(?:Mob|Ph)[\s:]*(\d{10})'
                ],
                keywords=['mobile', 'phone', 'contact', 'number'],
                examples=['+91 9876543210', 'Mobile: 9876543210'],
                description='Mobile number field detection'
            ),
            
            'CONTACT_EMAIL': FieldCategory(
                category_name='email',
                category_type='contact',
                field_names=['email', 'email_id', 'email_address'],
                patterns=[
                    r'(?:Email|E-mail)[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
                ],
                keywords=['email', 'e-mail', '@', 'mail'],
                examples=['john@example.com', 'Email: user@domain.com'],
                description='Email address field detection'
            ),
            
            # Financial Fields
            'FINANCIAL_ACCOUNT': FieldCategory(
                category_name='account_number',
                category_type='financial',
                field_names=['account_number', 'bank_account', 'acc_no'],
                patterns=[
                    r'(?:Account|A/c|Acc)[\s:]*(?:No|Number)[\s:]*(\d{9,18})',
                    r'Bank\s*Account[\s:]*(\d{9,18})'
                ],
                keywords=['account number', 'bank account', 'acc no'],
                examples=['123456789012', 'Account No: 987654321'],
                description='Bank account number field detection'
            ),
            
            'FINANCIAL_IFSC': FieldCategory(
                category_name='ifsc_code',
                category_type='financial',
                field_names=['ifsc', 'ifsc_code', 'bank_code'],
                patterns=[
                    r'(?:IFSC|Ifsc)[\s:]*([A-Z]{4}0[A-Z0-9]{6})',
                    r'([A-Z]{4}0[A-Z0-9]{6})'
                ],
                keywords=['ifsc', 'ifsc code', 'bank code'],
                examples=['SBIN0001234', 'IFSC: HDFC0001234'],
                description='IFSC code field detection'
            )
        }
        
        print(f"✅ Initialized {len(categories)} field categories")
        return categories
    
    def generate_training_data(self) -> Tuple[List[str], List[str], List[str]]:
        """Generate comprehensive training data from field categories."""
        texts = []
        field_labels = []
        category_labels = []
        
        print("🔄 Generating training data from field categories...")
        
        for category_name, category in self.field_categories.items():
            # Add examples as training texts
            for example in category.examples:
                texts.append(example.lower())
                field_labels.append(category.category_name)
                category_labels.append(category.category_type)
            
            # Generate synthetic texts from patterns and keywords
            for keyword in category.keywords:
                for field_name in category.field_names:
                    synthetic_text = f"{keyword} {field_name} identification document"
                    texts.append(synthetic_text.lower())
                    field_labels.append(category.category_name)
                    category_labels.append(category.category_type)
            
            # Add pattern-based synthetic data
            for pattern in category.patterns:
                # Simplified pattern to text conversion
                pattern_text = pattern.replace(r'(?:', '').replace(')', '').replace('[', '').replace(']', '')
                pattern_text = pattern_text.replace('\\s*', ' ').replace('\\d', 'digit').replace('+', '')
                pattern_text = f"document contains {pattern_text} pattern field"
                texts.append(pattern_text.lower())
                field_labels.append(category.category_name)
                category_labels.append(category.category_type)
        
        print(f"📊 Generated {len(texts)} training samples across {len(set(field_labels))} field types")
        print(f"📈 Category distribution: {dict(zip(*np.unique(category_labels, return_counts=True)))}")
        
        return texts, field_labels, category_labels
    
    def train_field_detection_models(self) -> Dict[str, Any]:
        """Train multiple models for field detection and categorization."""
        print("\n🤖 Training Field Detection Models...")
        print("=" * 50)
        
        # Generate training data
        texts, field_labels, category_labels = self.generate_training_data()
        
        if not texts:
            raise ValueError("No training data generated")
        
        # Initialize results dictionary
        results = {
            'field_models': {},
            'category_models': {},
            'best_field_model': None,
            'best_category_model': None,
            'training_info': {
                'samples': len(texts),
                'unique_fields': len(set(field_labels)),
                'unique_categories': len(set(category_labels)),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Prepare data splits
        X_train, X_test, y_field_train, y_field_test, y_cat_train, y_cat_test = train_test_split(
            texts, field_labels, category_labels, test_size=0.2, random_state=42, stratify=field_labels
        )
        
        # Train field-level classification models
        print("\n🎯 Training Field-Level Classification Models...")
        field_models = self._train_classification_models(
            X_train, X_test, y_field_train, y_field_test, "Field"
        )
        results['field_models'] = field_models
        
        # Train category-level classification models
        print("\n🏷️ Training Category-Level Classification Models...")
        category_models = self._train_classification_models(
            X_train, X_test, y_cat_train, y_cat_test, "Category"
        )
        results['category_models'] = category_models
        
        # Select best models based on F1 score
        best_field_f1 = 0
        best_category_f1 = 0
        
        for model_name, model_info in field_models.items():
            if model_info['f1_score'] > best_field_f1:
                best_field_f1 = model_info['f1_score']
                results['best_field_model'] = model_name
                self.field_classifier = model_info['model']
        
        for model_name, model_info in category_models.items():
            if model_info['f1_score'] > best_category_f1:
                best_category_f1 = model_info['f1_score']
                results['best_category_model'] = model_name
                self.category_classifier = model_info['model']
        
        print(f"\n✅ Best Field Model: {results['best_field_model']} (F1: {best_field_f1:.3f})")
        print(f"✅ Best Category Model: {results['best_category_model']} (F1: {best_category_f1:.3f})")
        
        return results\n    \n    def _train_classification_models(self, X_train: List[str], X_test: List[str], \n                                   y_train: List[str], y_test: List[str],\n                                   task_name: str) -> Dict[str, Any]:\n        \"\"\"Train multiple classification models for a specific task.\"\"\"\n        models = {\n            'naive_bayes': Pipeline([\n                ('tfidf', TfidfVectorizer(\n                    max_features=1000,\n                    ngram_range=(1, 2),\n                    min_df=1,\n                    max_df=0.95\n                )),\n                ('nb', MultinomialNB(alpha=0.1))\n            ]),\n            \n            'random_forest': Pipeline([\n                ('tfidf', TfidfVectorizer(\n                    max_features=1000,\n                    ngram_range=(1, 2),\n                    min_df=1,\n                    max_df=0.95\n                )),\n                ('rf', RandomForestClassifier(\n                    n_estimators=100,\n                    random_state=42,\n                    max_depth=10\n                ))\n            ]),\n            \n            'svm': Pipeline([\n                ('tfidf', TfidfVectorizer(\n                    max_features=1000,\n                    ngram_range=(1, 2),\n                    min_df=1,\n                    max_df=0.95\n                )),\n                ('svm', SVC(\n                    kernel='linear',\n                    probability=True,\n                    random_state=42\n                ))\n            ]),\n            \n            'logistic_regression': Pipeline([\n                ('tfidf', TfidfVectorizer(\n                    max_features=1000,\n                    ngram_range=(1, 2),\n                    min_df=1,\n                    max_df=0.95\n                )),\n                ('lr', LogisticRegression(\n                    random_state=42,\n                    max_iter=1000\n                ))\n            ])\n        }\n        \n        results = {}\n        \n        for model_name, model in models.items():\n            print(f\"  🔄 Training {model_name} for {task_name}...\")\n            \n            try:\n                # Train the model\n                model.fit(X_train, y_train)\n                \n                # Make predictions\n                y_pred = model.predict(X_test)\n                \n                # Calculate metrics\n                accuracy = accuracy_score(y_test, y_pred)\n                f1 = f1_score(y_test, y_pred, average='weighted')\n                \n                # Cross-validation\n                cv_scores = cross_val_score(model, X_train + X_test, y_train + y_test, \n                                          cv=3, scoring='f1_weighted')\n                \n                results[model_name] = {\n                    'model': model,\n                    'accuracy': accuracy,\n                    'f1_score': f1,\n                    'cv_mean': cv_scores.mean(),\n                    'cv_std': cv_scores.std(),\n                    'predictions': y_pred\n                }\n                \n                print(f\"    ✅ {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}\")\n                \n            except Exception as e:\n                print(f\"    ❌ {model_name} failed: {e}\")\n                continue\n        \n        return results\n    \n    def detect_fields_in_text(self, text: str, confidence_threshold: float = 0.5) -> List[FieldDetectionResult]:\n        \"\"\"Detect fields in OCR text using trained models.\"\"\"\n        if not self.field_classifier or not self.category_classifier:\n            raise ValueError(\"Models not trained. Please train models first.\")\n        \n        results = []\n        text_lower = text.lower()\n        \n        # Split text into potential field segments\n        segments = self._segment_text_for_field_detection(text)\n        \n        for segment in segments:\n            try:\n                # Predict field type\n                field_proba = self.field_classifier.predict_proba([segment.lower()])[0]\n                field_classes = self.field_classifier.classes_\n                field_idx = np.argmax(field_proba)\n                field_confidence = field_proba[field_idx]\n                \n                if field_confidence < confidence_threshold:\n                    continue\n                \n                field_name = field_classes[field_idx]\n                \n                # Predict category type\n                category_proba = self.category_classifier.predict_proba([segment.lower()])[0]\n                category_classes = self.category_classifier.classes_\n                category_idx = np.argmax(category_proba)\n                category_confidence = category_proba[category_idx]\n                category_type = category_classes[category_idx]\n                \n                # Extract actual field value using patterns\n                detected_value, pattern_matched = self._extract_field_value_from_segment(\n                    segment, field_name\n                )\n                \n                # Create result\n                result = FieldDetectionResult(\n                    field_name=field_name,\n                    field_category=field_name,\n                    category_type=category_type,\n                    detected_value=detected_value,\n                    confidence=min(field_confidence, category_confidence),\n                    context=segment,\n                    pattern_matched=pattern_matched,\n                    reasoning=f\"Field: {field_name} ({field_confidence:.3f}), Category: {category_type} ({category_confidence:.3f})\"\n                )\n                \n                results.append(result)\n                \n            except Exception as e:\n                print(f\"Warning: Error processing segment '{segment[:50]}...': {e}\")\n                continue\n        \n        # Sort by confidence and remove duplicates\n        results.sort(key=lambda x: x.confidence, reverse=True)\n        unique_results = []\n        seen_fields = set()\n        \n        for result in results:\n            if result.field_name not in seen_fields:\n                unique_results.append(result)\n                seen_fields.add(result.field_name)\n        \n        return unique_results\n    \n    def _segment_text_for_field_detection(self, text: str) -> List[str]:\n        \"\"\"Segment text into potential field-containing chunks.\"\"\"\n        # Split by lines first\n        lines = text.strip().split('\\n')\n        segments = []\n        \n        for line in lines:\n            line = line.strip()\n            if not line:\n                continue\n            \n            # Add full line as segment\n            segments.append(line)\n            \n            # Also add individual words/phrases that might contain field info\n            words = line.split()\n            if len(words) > 1:\n                # Create overlapping windows\n                for i in range(len(words)):\n                    for j in range(i + 2, min(i + 6, len(words) + 1)):\n                        segment = ' '.join(words[i:j])\n                        segments.append(segment)\n        \n        return list(set(segments))  # Remove duplicates\n    \n    def _extract_field_value_from_segment(self, segment: str, field_name: str) -> Tuple[Optional[str], str]:\n        \"\"\"Extract the actual field value from a text segment.\"\"\"\n        # Find the corresponding field category\n        field_category = None\n        for cat_name, category in self.field_categories.items():\n            if category.category_name == field_name:\n                field_category = category\n                break\n        \n        if not field_category:\n            return None, \"No pattern\"\n        \n        # Try to match patterns\n        for pattern in field_category.patterns:\n            try:\n                import re\n                matches = re.findall(pattern, segment, re.IGNORECASE)\n                if matches:\n                    if isinstance(matches[0], tuple):\n                        return matches[0][0], pattern\n                    return matches[0], pattern\n            except Exception:\n                continue\n        \n        # If no pattern match, return the segment itself if it contains keywords\n        for keyword in field_category.keywords:\n            if keyword.lower() in segment.lower():\n                return segment.strip(), f\"keyword: {keyword}\"\n        \n        return None, \"No match\"\n    \n    def save_trained_models(self, base_path: str = None) -> Dict[str, str]:\n        \"\"\"Save trained models to disk.\"\"\"\n        if not base_path:\n            base_path = self.models_dir\n        \n        base_path = Path(base_path)\n        base_path.mkdir(parents=True, exist_ok=True)\n        \n        saved_paths = {}\n        \n        # Save field classifier\n        if self.field_classifier:\n            field_path = base_path / \"field_classifier.pkl\"\n            with open(field_path, 'wb') as f:\n                pickle.dump(self.field_classifier, f)\n            saved_paths['field_classifier'] = str(field_path)\n        \n        # Save category classifier\n        if self.category_classifier:\n            category_path = base_path / \"category_classifier.pkl\"\n            with open(category_path, 'wb') as f:\n                pickle.dump(self.category_classifier, f)\n            saved_paths['category_classifier'] = str(category_path)\n        \n        # Save field categories\n        categories_path = base_path / \"field_categories.json\"\n        categories_dict = {}\n        for name, category in self.field_categories.items():\n            categories_dict[name] = {\n                'category_name': category.category_name,\n                'category_type': category.category_type,\n                'field_names': category.field_names,\n                'patterns': category.patterns,\n                'keywords': category.keywords,\n                'examples': category.examples,\n                'description': category.description,\n                'confidence_threshold': category.confidence_threshold\n            }\n        \n        with open(categories_path, 'w') as f:\n            json.dump(categories_dict, f, indent=2)\n        saved_paths['field_categories'] = str(categories_path)\n        \n        print(f\"💾 Models saved to: {base_path}\")\n        return saved_paths\n    \n    def load_trained_models(self, base_path: str = None) -> bool:\n        \"\"\"Load trained models from disk.\"\"\"\n        if not base_path:\n            base_path = self.models_dir\n        \n        base_path = Path(base_path)\n        \n        try:\n            # Load field classifier\n            field_path = base_path / \"field_classifier.pkl\"\n            if field_path.exists():\n                with open(field_path, 'rb') as f:\n                    self.field_classifier = pickle.load(f)\n            \n            # Load category classifier\n            category_path = base_path / \"category_classifier.pkl\"\n            if category_path.exists():\n                with open(category_path, 'rb') as f:\n                    self.category_classifier = pickle.load(f)\n            \n            # Load field categories\n            categories_path = base_path / \"field_categories.json\"\n            if categories_path.exists():\n                with open(categories_path, 'r') as f:\n                    categories_dict = json.load(f)\n                \n                self.field_categories = {}\n                for name, data in categories_dict.items():\n                    self.field_categories[name] = FieldCategory(**data)\n            \n            print(f\"📥 Models loaded from: {base_path}\")\n            return True\n            \n        except Exception as e:\n            print(f\"❌ Error loading models: {e}\")\n            return False\n    \n    def generate_training_report(self, results: Dict[str, Any]) -> str:\n        \"\"\"Generate comprehensive training report.\"\"\"\n        report = []\n        report.append(\"🎯 FIELD DETECTION MODEL TRAINING REPORT\")\n        report.append(\"=\" * 60)\n        report.append(f\"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")\n        report.append(f\"\")\n        \n        # Training info\n        info = results['training_info']\n        report.append(f\"📊 TRAINING DATASET SUMMARY\")\n        report.append(f\"-\" * 30)\n        report.append(f\"Total Samples: {info['samples']}\")\n        report.append(f\"Unique Fields: {info['unique_fields']}\")\n        report.append(f\"Unique Categories: {info['unique_categories']}\")\n        report.append(f\"\")\n        \n        # Field models performance\n        report.append(f\"🎯 FIELD CLASSIFICATION MODELS\")\n        report.append(f\"-\" * 35)\n        for model_name, model_info in results['field_models'].items():\n            report.append(f\"{model_name.replace('_', ' ').title()}:\")\n            report.append(f\"  Accuracy: {model_info['accuracy']:.3f}\")\n            report.append(f\"  F1 Score: {model_info['f1_score']:.3f}\")\n            report.append(f\"  CV Mean: {model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}\")\n        \n        report.append(f\"\")\n        \n        # Category models performance\n        report.append(f\"🏷️ CATEGORY CLASSIFICATION MODELS\")\n        report.append(f\"-\" * 37)\n        for model_name, model_info in results['category_models'].items():\n            report.append(f\"{model_name.replace('_', ' ').title()}:\")\n            report.append(f\"  Accuracy: {model_info['accuracy']:.3f}\")\n            report.append(f\"  F1 Score: {model_info['f1_score']:.3f}\")\n            report.append(f\"  CV Mean: {model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}\")\n        \n        report.append(f\"\")\n        report.append(f\"✅ BEST MODELS SELECTED\")\n        report.append(f\"-\" * 25)\n        report.append(f\"Best Field Model: {results['best_field_model']}\")\n        report.append(f\"Best Category Model: {results['best_category_model']}\")\n        \n        return \"\\n\".join(report)\n\ndef main():\n    \"\"\"Main function to demonstrate field detection training.\"\"\"\n    print(\"🎯 Field Detection Model Training for OCR Text\")\n    print(\"=\" * 55)\n    \n    # Initialize field classifier\n    classifier = FieldCategoryClassifier()\n    \n    try:\n        # Train the models\n        print(\"\\n🚀 Starting Model Training...\")\n        results = classifier.train_field_detection_models()\n        \n        # Save the trained models\n        print(\"\\n💾 Saving Trained Models...\")\n        saved_paths = classifier.save_trained_models()\n        \n        # Generate training report\n        print(\"\\n📊 Generating Training Report...\")\n        report = classifier.generate_training_report(results)\n        print(\"\\n\" + report)\n        \n        # Save report to file\n        report_path = Path(\"models/field_detection/training_report.txt\")\n        with open(report_path, 'w') as f:\n            f.write(report)\n        \n        # Test with sample OCR text\n        print(\"\\n🧪 Testing Field Detection with Sample OCR Text...\")\n        print(\"-\" * 55)\n        \n        sample_texts = [\n            \"Name: RAJESH KUMAR\\nS/O: RAM PRASAD\\nDOB: 15/08/1990\\nAadhar: 1234 5678 9012\",\n            \"PAN: ABCDE1234F\\nMobile: +91 9876543210\\nEmail: rajesh@example.com\",\n            \"Address: 123 MG Road, Bangalore\\nPIN: 560001\\nVoter ID: ABC1234567\"\n        ]\n        \n        for i, sample_text in enumerate(sample_texts, 1):\n            print(f\"\\nSample {i}:\")\n            print(f\"Text: {sample_text}\")\n            \n            detected_fields = classifier.detect_fields_in_text(sample_text)\n            \n            if detected_fields:\n                print(f\"Detected Fields:\")\n                for field in detected_fields:\n                    print(f\"  • {field.field_name} ({field.category_type}): '{field.detected_value}' (conf: {field.confidence:.3f})\")\n            else:\n                print(\"  No fields detected\")\n        \n        print(f\"\\n✅ Field Detection Training Complete!\")\n        print(f\"💾 Models saved to: {saved_paths['field_classifier']}\")\n        print(f\"📊 Report saved to: {report_path}\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ Training failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\nif __name__ == \"__main__\":\n    success = main()\n    if not success:\n        sys.exit(1)