#!/usr/bin/env python3
"""
Field Extraction Pipeline
Integrates field detection model with OCR processing to extract and categorize fields.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

from field_detection_model_new import FieldDetectionModel
from ocr.ocr import extract_text
from ocr.preprocess import AdvancedImagePreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractedField:
    """Represents a single extracted field."""
    def __init__(self, field_name: str, field_category: str, category_type: str, 
                 value: str, confidence: float, extraction_method: str = "AI", 
                 validation_status: str = "pending", context: str = ""):
        self.field_name = field_name
        self.field_category = field_category
        self.category_type = category_type
        self.value = value
        self.confidence = confidence
        self.extraction_method = extraction_method
        self.validation_status = validation_status
        self.context = context

class FieldAnalysisResult:
    """Analysis result compatible with app.py expectations."""
    def __init__(self, extracted_fields: List[ExtractedField], document_type: str = "general",
                 document_confidence: float = 0.8, processing_time: float = 0.0,
                 extraction_timestamp: str = None):
        self.extracted_fields = extracted_fields
        self.document_type = document_type
        self.document_confidence = document_confidence
        self.processing_time = processing_time
        self.extraction_timestamp = extraction_timestamp or datetime.now().isoformat()

class FieldExtractionPipeline:
    """
    Complete pipeline for extracting and categorizing fields from OCR text.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the field extraction pipeline."""
        self.field_model = FieldDetectionModel()
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.field_model.load_model(model_path)
            logger.info(f"✅ Loaded pre-trained field detection model from {model_path}")
        else:
            # Train new model
            self._train_field_model()
        
        logger.info("✅ FieldExtractionPipeline initialized")
    
    def _train_field_model(self):
        """Train the field detection model."""
        try:
            logger.info("🔄 Training field detection model...")
            results = self.field_model.train('random_forest')
            logger.info(f"✅ Model trained with accuracy: {results['accuracy']:.3f}")
        except Exception as e:
            logger.error(f"❌ Failed to train field detection model: {e}")
    
    def extract_fields_from_image(self, image_path: str, 
                                 preprocessing_strategy: str = "adaptive_threshold") -> Dict[str, Any]:
        """
        Extract and categorize fields from an image using OCR and field detection.
        
        Args:
            image_path: Path to the image file
            preprocessing_strategy: Image preprocessing strategy to use
            
        Returns:
            Dictionary containing extracted text, detected fields, and categories
        """
        try:
            # Preprocess image
            logger.info(f"📸 Preprocessing image: {image_path}")
            processed_image = self.preprocessor.preprocess(image_path, preprocessing_strategy)
            
            # Extract text using OCR
            logger.info("🔍 Extracting text from image...")
            extracted_text = extract_text(processed_image)
            
            if not extracted_text or len(extracted_text.strip()) < 5:
                return {
                    'success': False,
                    'error': 'No text extracted from image',
                    'image_path': image_path
                }
            
            # Process the extracted text to find fields
            field_results = self.extract_fields_from_text(extracted_text)
            
            # Combine results
            result = {
                'success': True,
                'image_path': image_path,
                'preprocessing_strategy': preprocessing_strategy,
                'extracted_text': extracted_text,
                'text_confidence': 0.8,  # Default confidence
                'field_detection_results': field_results,
                'processing_time': 0,  # Default processing time
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Successfully extracted {len(field_results['detected_fields'])} fields from image")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def extract_fields_from_text(self, text: str, document_image_path: str = None) -> Dict[str, Any]:
        """
        Extract and categorize fields from plain text.
        
        Args:
            text: Input text to analyze
            document_image_path: Optional path to the source document image (for future enhancements)
            
        Returns:
            Dictionary containing detected fields and their categories
        """
        try:
            # Split text into potential fields
            potential_fields = self._extract_potential_fields(text)
            
            if not potential_fields:
                return {
                    'detected_fields': [],
                    'field_categories': {},
                    'field_count_by_category': {},
                    'total_fields': 0
                }
            
            # Classify each potential field
            extracted_fields_list = []
            field_categories = {}
            field_names_seen = set()  # Track field names to avoid duplicates
            
            for field_text in potential_fields:
                if len(field_text.strip()) > 1:  # Skip very short fields
                    try:
                        prediction = self.field_model.predict(field_text, return_probabilities=True)
                        
                        # Create unique field identifier to avoid duplicates
                        field_key = f"{field_text}_{prediction['predicted_category']}"
                        if field_key in field_names_seen:
                            continue  # Skip duplicate field
                        field_names_seen.add(field_key)
                        
                        # Create ExtractedField object
                        extracted_field = ExtractedField(
                            field_name=field_text.strip(),
                            field_category=prediction['predicted_category'],
                            category_type=prediction['predicted_category'],  # Using same for now
                            value=field_text.strip(),
                            confidence=prediction.get('confidence', 0.0),
                            extraction_method="AI Field Detection",
                            validation_status="detected",
                            context=text[:100] + "..." if len(text) > 100 else text
                        )
                        
                        extracted_fields_list.append(extracted_field)
                        
                        # Group by category for backward compatibility
                        category = prediction['predicted_category']
                        if category not in field_categories:
                            field_categories[category] = []
                        field_categories[category].append({
                            'field_text': field_text,
                            'predicted_category': prediction['predicted_category'],
                            'confidence': prediction.get('confidence', 0),
                            'probabilities': prediction.get('probabilities', {}),
                            'processed_text': prediction.get('processed_text', ''),
                            'field_id': len(extracted_fields_list)
                        })
                    
                    except Exception as e:
                        logger.warning(f"Error processing field '{field_text}': {e}")
                        continue
            
            # Create analysis result
            analysis_result = FieldAnalysisResult(
                extracted_fields=extracted_fields_list,
                document_type="general",
                document_confidence=0.8,
                processing_time=0.0
            )
            
            logger.info(f"✅ Detected {len(extracted_fields_list)} fields across {len(field_categories)} categories")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error extracting fields from text: {e}")
            # Return empty analysis result on error
            return FieldAnalysisResult(
                extracted_fields=[],
                document_type="general",
                document_confidence=0.0,
                processing_time=0.0
            )
    
    def _extract_potential_fields(self, text: str) -> List[str]:
        """
        Extract potential field names/labels from text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential field strings
        """
        potential_fields = []
        
        # Split by lines first
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns like "Label:" or "Label :"
            colon_pattern_fields = self._extract_colon_pattern_fields(line)
            potential_fields.extend(colon_pattern_fields)
            
            # Look for standalone words that might be field labels
            standalone_fields = self._extract_standalone_field_candidates(line)
            potential_fields.extend(standalone_fields)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fields = []
        for field in potential_fields:
            field_clean = field.strip().lower()
            if field_clean not in seen and len(field_clean) > 1:
                seen.add(field_clean)
                unique_fields.append(field.strip())
        
        return unique_fields
    
    def _extract_colon_pattern_fields(self, line: str) -> List[str]:
        """Extract fields that follow the 'Label:' or 'Label :' pattern."""
        import re
        
        # Pattern to match "word(s):" or "word(s) :"
        pattern = r'([A-Za-z][A-Za-z\s]*?)\s*:'
        matches = re.findall(pattern, line)
        
        fields = []
        for match in matches:
            field = match.strip()
            if len(field) > 1 and not field.lower() in ['http', 'https', 'ftp']:
                fields.append(field)
        
        return fields
    
    def _extract_standalone_field_candidates(self, line: str) -> List[str]:
        """Extract standalone words that might be field labels."""
        import re
        
        # Skip lines that are too long (likely to be content, not labels)
        if len(line) > 50:
            return []
        
        # Look for capitalized words or phrases
        words = line.split()
        
        candidates = []
        for i, word in enumerate(words):
            # Single capitalized words
            if word[0].isupper() and len(word) > 2:
                candidates.append(word)
            
            # Two-word phrases starting with capital
            if i < len(words) - 1 and word[0].isupper() and words[i+1][0].isupper():
                phrase = f"{word} {words[i+1]}"
                if len(phrase) <= 30:  # Reasonable field name length
                    candidates.append(phrase)
        
        return candidates
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the overall structure of a document based on detected fields.
        
        Args:
            text: Document text
            
        Returns:
            Document structure analysis
        """
        field_results = self.extract_fields_from_text(text)
        
        if not field_results['detected_fields']:
            return {
                'document_type': 'unknown',
                'structure_analysis': 'No fields detected',
                'confidence': 0.0
            }
        
        categories = field_results['field_count_by_category']
        
        # Determine document type based on field categories
        document_type = 'unknown'
        confidence = 0.0
        
        # Academic document detection
        if categories.get('academic', 0) >= 2:
            document_type = 'academic_certificate'
            confidence = 0.8 + min(categories.get('academic', 0) * 0.1, 0.2)
        
        # Identification document detection
        elif categories.get('identification', 0) >= 1 and categories.get('personal_info', 0) >= 2:
            document_type = 'identification_document'
            confidence = 0.7 + min(categories.get('identification', 0) * 0.15, 0.3)
        
        # Financial document detection
        elif categories.get('financial', 0) >= 2:
            document_type = 'financial_document'
            confidence = 0.7 + min(categories.get('financial', 0) * 0.1, 0.3)
        
        # Personal information form
        elif categories.get('personal_info', 0) >= 3:
            document_type = 'personal_information_form'
            confidence = 0.6 + min(categories.get('personal_info', 0) * 0.1, 0.4)
        
        # General form
        elif field_results['total_fields'] >= 3:
            document_type = 'general_form'
            confidence = min(field_results['total_fields'] * 0.1, 0.6)
        
        structure_analysis = {
            'total_fields': field_results['total_fields'],
            'category_distribution': categories,
            'dominant_categories': sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3],
            'field_density': field_results['total_fields'] / max(len(text.split('\n')), 1)
        }
        
        return {
            'document_type': document_type,
            'confidence': confidence,
            'structure_analysis': structure_analysis,
            'field_results': field_results
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the field detection model."""
        return self.field_model.get_model_info()
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save extraction results to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save results to {output_path}: {e}")
            return False


def main():
    """Demo function to test field extraction pipeline."""
    print("🔍 Field Extraction Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = FieldExtractionPipeline()
    
    # Test with sample text
    sample_text = """
    Student Certificate
    
    Full Name: John Doe
    Student ID: STU123456
    Course: Computer Science
    Grade: A
    University: Tech University
    Issue Date: 2023-11-15
    Certificate Number: CERT789
    """
    
    print("📄 Testing with sample text:")
    print(sample_text)
    print("\n" + "-" * 30)
    
    # Extract fields
    results = pipeline.extract_fields_from_text(sample_text)
    
    print(f"🔍 Field Detection Results:")
    print(f"   Total fields detected: {results['total_fields']}")
    print(f"   Categories found: {', '.join(results['unique_categories'])}")
    
    print(f"\n📊 Fields by category:")
    for category, count in results['field_count_by_category'].items():
        print(f"   {category}: {count} fields")
    
    print(f"\n📝 Detected fields:")
    for field in results['detected_fields']:
        print(f"   '{field['field_text']}' → {field['predicted_category']} "
              f"(confidence: {field['confidence']:.3f})")
    
    # Analyze document structure
    print(f"\n📋 Document structure analysis:")
    structure = pipeline.analyze_document_structure(sample_text)
    print(f"   Document type: {structure['document_type']}")
    print(f"   Confidence: {structure['confidence']:.3f}")
    
    print(f"\n🎉 Field extraction pipeline demo completed!")


if __name__ == "__main__":
    main()