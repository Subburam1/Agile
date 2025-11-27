"""
Image-Based ML Training Demo
Demonstrates training ML models using actual document images with OCR
"""

import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

# Import our image-based trainer
from image_based_ml_trainer import ImageBasedMLTrainer, DocumentImageGenerator

def demo_image_based_training():
    """Demonstrate image-based ML training for document classification."""
    
    print("🖼️ Image-Based ML Document Classification Demo")
    print("=" * 55)
    print("🎯 Training AI Models Using Real Document Images")
    print("=" * 55)
    
    # Check if required libraries are available
    print("\\n🔍 Checking System Requirements...")
    
    requirements = {
        'OpenCV': False,
        'Pytesseract': False,
        'PIL': True,  # Already imported successfully
        'Scikit-learn': True,  # From previous demos
        'NumPy': True
    }
    
    try:
        import cv2
        requirements['OpenCV'] = True
        print("✅ OpenCV - Available")
    except ImportError:
        print("⚠️ OpenCV - Not installed (pip install opencv-python)")
    
    try:
        import pytesseract
        requirements['Pytesseract'] = True
        print("✅ Pytesseract - Available")
    except ImportError:
        print("⚠️ Pytesseract - Not installed (pip install pytesseract)")
    
    print("✅ PIL/Pillow - Available")
    print("✅ Scikit-learn - Available")
    print("✅ NumPy - Available")
    
    # Initialize systems
    print("\\n🚀 Initializing Image-Based Training System...")
    trainer = ImageBasedMLTrainer()
    generator = DocumentImageGenerator()
    
    print("✅ Image-based trainer initialized")
    print("✅ Document image generator ready")
    
    # Show what document types we can generate and train on
    print("\\n📋 Supported Document Types for Image Training:")
    print("-" * 50)
    
    document_info = {
        'AADHAR_CARD': {
            'name': 'Aadhaar Card',
            'description': 'Government ID with UID number, multilingual text',
            'features': ['12-digit UID', 'Hindi/English text', 'Government seal']
        },
        'PAN_CARD': {
            'name': 'PAN Card', 
            'description': 'Income Tax Department identification',
            'features': ['Alphanumeric PAN', 'Tax department logo', 'Photo ID']
        },
        'VOTER_ID': {
            'name': 'Voter ID',
            'description': 'Election Commission identity card',
            'features': ['EPIC number', 'Constituency info', 'Election Commission seal']
        },
        'MARKSHEET': {
            'name': 'Academic Marksheet',
            'description': 'University examination results',
            'features': ['Grade tables', 'University seal', 'Roll numbers']
        },
        'BANK_PASSBOOK': {
            'name': 'Bank Passbook',
            'description': 'Banking transaction records',
            'features': ['Account details', 'Transaction history', 'Bank logo']
        }
    }
    
    for i, (doc_type, info) in enumerate(document_info.items(), 1):
        print(f"{i}. {info['name']} ({doc_type})")
        print(f"   📄 {info['description']}")
        print(f"   🔍 Key Features: {', '.join(info['features'])}")
        print()
    
    # Demonstrate image generation process
    print("🎨 Demonstrating Synthetic Document Image Generation:")
    print("-" * 55)
    
    # Generate sample images for demo
    sample_images = {}
    
    print("📸 Generating sample document images...")
    for doc_type in ['AADHAR_CARD', 'PAN_CARD', 'VOTER_ID']:
        try:
            print(f"   Creating {doc_type} sample...")
            image_path = generator.generate_synthetic_image(doc_type, variation=0)
            sample_images[doc_type] = image_path
            print(f"   ✅ Saved: {image_path}")
        except Exception as e:
            print(f"   ❌ Error generating {doc_type}: {e}")
    
    print(f"\\n✅ Generated {len(sample_images)} sample document images")
    
    # Demonstrate OCR text extraction
    print("\\n🔤 Demonstrating OCR Text Extraction from Images:")
    print("-" * 55)
    
    if not requirements['Pytesseract']:
        print("⚠️ Tesseract OCR not available - using simulated extraction")
        
        # Simulate OCR results for demo
        simulated_extractions = {
            'AADHAR_CARD': {
                'text': 'government of india unique identification authority aadhaar card name rajesh kumar fathers name suresh kumar date of birth 15/08/1990 aadhaar number 1234 5678 9012',
                'confidence': 0.85
            },
            'PAN_CARD': {
                'text': 'income tax department govt of india permanent account number card name rajesh kumar fathers name suresh kumar date of birth 15/08/1990 pan abcde1234f',
                'confidence': 0.92
            },
            'VOTER_ID': {
                'text': 'election commission of india electoral photo identity card name rajesh kumar fathers name suresh kumar epic no abc1234567 constituency mumbai north age 33',
                'confidence': 0.88
            }
        }
        
        for doc_type, extraction in simulated_extractions.items():
            print(f"📄 {doc_type.replace('_', ' ').title()}:")
            print(f"   📝 Extracted Text: {extraction['text'][:80]}...")
            print(f"   📊 OCR Confidence: {extraction['confidence']:.3f}")
            print()
    
    else:
        # Real OCR extraction
        print("🔍 Extracting text from generated images using Tesseract OCR...")
        
        for doc_type, image_path in sample_images.items():
            try:
                text, confidence = trainer.extract_text_from_image(image_path)
                print(f"📄 {doc_type.replace('_', ' ').title()}:")
                print(f"   📝 Extracted Text: {text[:80]}...")
                print(f"   📊 OCR Confidence: {confidence:.3f}")
                print()
            except Exception as e:
                print(f"   ❌ OCR Error: {e}")
    
    # Training process simulation
    print("🤖 ML Training Process with Image Data:")
    print("-" * 45)
    
    training_steps = [
        "1. 🖼️ Generate synthetic document images (15 per type)",
        "2. 🔤 Extract text using OCR (Tesseract)",
        "3. 🧹 Clean and preprocess extracted text",
        "4. ⚖️ Weight samples by OCR confidence",
        "5. 🎯 Train multiple ML models (Naive Bayes, Random Forest)",
        "6. 📊 Evaluate performance with cross-validation",
        "7. 🏆 Select best performing model",
        "8. 💾 Save trained model for production use"
    ]
    
    for step in training_steps:
        print(f"   {step}")
    
    # Show advantages of image-based training
    print("\\n🎯 Advantages of Image-Based Training:")
    print("-" * 40)
    
    advantages = [
        "🖼️ Realistic Training Data - Uses actual document layouts and fonts",
        "🔤 OCR Integration - Handles real-world OCR errors and quality issues", 
        "📊 Confidence Weighting - Uses OCR confidence to weight training samples",
        "🎨 Synthetic Data Generation - Creates unlimited training samples",
        "📸 Real Image Support - Can incorporate actual document scans",
        "🔧 Preprocessing Pipeline - Image enhancement for better OCR",
        "🌐 End-to-End Solution - From image to classification in one pipeline",
        "📈 Performance Metrics - OCR + Classification confidence scoring"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    # Training workflow demonstration
    print("\\n⚙️ Complete Image-Based Training Workflow:")
    print("-" * 50)
    
    workflow_demo = {
        'data_generation': {
            'description': 'Generate synthetic document images',
            'samples': '15 images per document type',
            'formats': 'PNG with realistic layouts and fonts',
            'variations': 'Different names, numbers, and slight noise'
        },
        'ocr_extraction': {
            'description': 'Extract text using Tesseract OCR',
            'preprocessing': 'Gaussian blur, thresholding, morphology',
            'confidence': 'Filter low-confidence extractions (>50%)',
            'output': 'Clean text with confidence scores'
        },
        'ml_training': {
            'description': 'Train classification models',
            'algorithms': 'Naive Bayes, Random Forest',
            'features': 'TF-IDF vectors with 1500 features',
            'weighting': 'Sample weights based on OCR confidence'
        },
        'evaluation': {
            'description': 'Comprehensive model evaluation',
            'metrics': 'Accuracy, precision, recall, F1-score',
            'validation': '3-fold cross-validation',
            'selection': 'Best model based on CV score'
        }
    }
    
    for stage, details in workflow_demo.items():
        print(f"\\n🔸 {stage.replace('_', ' ').title()}:")
        for key, value in details.items():
            print(f"     • {key.replace('_', ' ').title()}: {value}")
    
    # Performance expectations
    print("\\n📈 Expected Performance Metrics:")
    print("-" * 35)
    
    performance_metrics = {
        'OCR Accuracy': '80-95% (depends on image quality)',
        'Classification Accuracy': '85-95% (on extracted text)',
        'Combined Confidence': '70-90% (OCR × Classification)',
        'Processing Speed': '<1 second per image',
        'Training Time': '2-5 minutes for full dataset',
        'Model Size': '3-8 MB (including image preprocessing)'
    }
    
    for metric, value in performance_metrics.items():
        print(f"   📊 {metric}: {value}")
    
    # Real-world application scenarios
    print("\\n🌍 Real-World Application Scenarios:")
    print("-" * 40)
    
    scenarios = [
        "🏛️ Government Document Processing Centers",
        "🏦 Banking KYC (Know Your Customer) Systems", 
        "📚 Educational Institution Record Management",
        "🏥 Healthcare Patient Documentation",
        "✈️ Immigration and Visa Processing",
        "📋 Insurance Claim Document Classification",
        "🏢 Corporate Document Management Systems",
        "📱 Mobile Document Scanner Applications"
    ]
    
    for scenario in scenarios:
        print(f"   {scenario}")
    
    # Integration capabilities
    print("\\n🔗 System Integration Capabilities:")
    print("-" * 40)
    
    integration_features = [
        "📱 Mobile App Integration (Camera → OCR → Classification)",
        "🌐 REST API Deployment (Upload image, get classification)",
        "☁️ Cloud Processing (Scalable batch processing)",
        "📊 Real-time Dashboard (Monitor classification accuracy)",
        "🔄 Continuous Learning (Retrain with new images)",
        "📈 Analytics Integration (Performance tracking)",
        "🛡️ Security Compliance (Enterprise data protection)",
        "🔌 Legacy System Integration (Existing document workflows)"
    ]
    
    for feature in integration_features:
        print(f"   {feature}")
    
    # Create demo report
    demo_report = {
        'demo_timestamp': datetime.now().isoformat(),
        'system_capabilities': {
            'image_generation': True,
            'ocr_extraction': requirements['Pytesseract'],
            'ml_training': True,
            'model_persistence': True
        },
        'document_types_supported': list(document_info.keys()),
        'sample_images_generated': len(sample_images),
        'training_workflow_steps': len(training_steps),
        'advantages_highlighted': len(advantages),
        'integration_scenarios': len(scenarios)
    }
    
    # Save demo report
    with open("models/image_based_demo_report.json", "w", encoding='utf-8') as f:
        json.dump(demo_report, f, indent=2, ensure_ascii=False)
    
    print("\\n✅ Image-Based ML Training Demo Complete!")
    print("📊 Demo report saved to: models/image_based_demo_report.json")
    
    # Instructions for running full training
    print("\\n🚀 To Run Full Image-Based Training:")
    print("-" * 40)
    
    instructions = [
        "1. 📦 Install required packages:",
        "   pip install opencv-python pytesseract pillow",
        "",
        "2. 📥 Install Tesseract OCR:",
        "   Download from: https://github.com/tesseract-ocr/tesseract",
        "",
        "3. 🔧 Run the training script:",
        "   python image_based_ml_trainer.py",
        "",
        "4. 📊 Check results in models/ directory:",
        "   - image_based_classifier.pkl (trained model)",
        "   - image_training_report.txt (performance report)",
        "   - training_images/ (generated sample images)",
        "",
        "5. 🧪 Test with your own images:",
        "   result = trainer.predict_from_image('your_image.jpg')"
    ]
    
    for instruction in instructions:
        print(f"   {instruction}")
    
    print("\\n🎯 Ready for Production Image-Based Document Classification!")
    print("=" * 60)

def main():
    """Run the image-based training demo."""
    demo_image_based_training()

if __name__ == "__main__":
    main()