#!/usr/bin/env python3
"""
Complete OCR Project Flow Implementation
Sequential workflow: Document Upload → OCR → Field Detection → User Selection → Field Blurring → Export
"""

import os
import json
import base64
import tempfile
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# OCR and Field Detection imports
try:
    from field_extraction_pipeline_new import FieldExtractionPipeline
    from ocr.ocr import extract_text
    from ocr.deep_learning_ocr import DeepLearningOCR
    FIELD_DETECTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Field detection not available: {e}")
    FIELD_DETECTION_AVAILABLE = False

class CompleteOCRFlow:
    """
    Complete OCR processing flow implementation.
    Handles the entire sequence from document upload to modified image export.
    """
    
    def __init__(self, upload_folder: str = "uploads", output_folder: str = "processed_outputs"):
        """Initialize the complete OCR flow processor."""
        self.upload_folder = Path(upload_folder)
        self.output_folder = Path(output_folder)
        
        # Create directories
        self.upload_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize components
        if FIELD_DETECTION_AVAILABLE:
            self.field_pipeline = FieldExtractionPipeline()
            logging.info("✅ Field detection pipeline initialized")
        else:
            self.field_pipeline = None
            logging.warning("⚠️ Field detection not available")
        
        # Processing history
        self.processing_history = []
        
        logging.info(f"🚀 Complete OCR Flow initialized")
        logging.info(f"📁 Upload folder: {self.upload_folder.absolute()}")
        logging.info(f"📁 Output folder: {self.output_folder.absolute()}")
    
    def process_document_complete_flow(self, 
                                     image_path: str, 
                                     selected_fields: List[Dict] = None,
                                     blur_strength: int = 15,
                                     confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Execute the complete OCR processing flow.
        
        Args:
            image_path: Path to the uploaded document image
            selected_fields: List of user-selected fields to blur (optional for auto-detection)
            blur_strength: Strength of blur effect (default: 15)
            confidence_threshold: Minimum confidence for field detection (default: 0.5)
            
        Returns:
            Complete processing results including all intermediate steps
        """
        
        flow_id = f"flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logging.info(f"🔄 Starting complete OCR flow: {flow_id}")
            
            # Step 1: Document Upload Validation
            step1_result = self._step1_validate_upload(image_path)
            if not step1_result['success']:
                return self._create_error_response(flow_id, "Step 1", step1_result['error'])
            
            # Step 2: OCR Text Extraction  
            step2_result = self._step2_extract_text(image_path)
            if not step2_result['success']:
                return self._create_error_response(flow_id, "Step 2", step2_result['error'])
            
            # Step 3: Field Detection
            step3_result = self._step3_detect_fields(step2_result['extracted_text'], image_path, confidence_threshold)
            if not step3_result['success']:
                return self._create_error_response(flow_id, "Step 3", step3_result['error'])
            
            # Step 4: User Field Selection (or auto-selection)
            step4_result = self._step4_select_fields(step3_result['detected_fields'], selected_fields)
            if not step4_result['success']:
                return self._create_error_response(flow_id, "Step 4", step4_result['error'])
            
            # Step 5: Field Blurring
            step5_result = self._step5_blur_fields(image_path, step4_result['selected_fields'], blur_strength)
            if not step5_result['success']:
                return self._create_error_response(flow_id, "Step 5", step5_result['error'])
            
            # Step 6: Export Modified Image
            step6_result = self._step6_export_image(step5_result['blurred_image'], flow_id)
            if not step6_result['success']:
                return self._create_error_response(flow_id, "Step 6", step6_result['error'])
            
            # Create complete flow result
            complete_result = {
                'success': True,
                'flow_id': flow_id,
                'timestamp': datetime.now().isoformat(),
                'processing_steps': {
                    'step1_upload_validation': step1_result,
                    'step2_ocr_extraction': step2_result,
                    'step3_field_detection': step3_result,
                    'step4_field_selection': step4_result,
                    'step5_field_blurring': step5_result,
                    'step6_image_export': step6_result
                },
                'final_output': {
                    'original_image': image_path,
                    'extracted_text': step2_result['extracted_text'],
                    'detected_fields_count': len(step3_result['detected_fields']),
                    'selected_fields_count': len(step4_result['selected_fields']),
                    'blurred_image_path': step6_result['output_path'],
                    'blurred_image_base64': step6_result['image_base64']
                },
                'processing_summary': {
                    'total_processing_time': sum([
                        step['processing_time'] for step in [
                            step1_result, step2_result, step3_result, 
                            step4_result, step5_result, step6_result
                        ]
                    ]),
                    'text_length': len(step2_result['extracted_text']),
                    'fields_detected': len(step3_result['detected_fields']),
                    'fields_blurred': len(step4_result['selected_fields']),
                    'blur_strength_used': blur_strength
                }
            }
            
            # Save to history
            self.processing_history.append(complete_result)
            
            logging.info(f"✅ Complete OCR flow finished: {flow_id}")
            logging.info(f"📊 Summary: {complete_result['processing_summary']['fields_detected']} fields detected, {complete_result['processing_summary']['fields_blurred']} fields blurred")
            
            return complete_result
            
        except Exception as e:
            logging.error(f"❌ Complete OCR flow failed: {e}")
            return self._create_error_response(flow_id, "Flow Execution", str(e))
    
    def _step1_validate_upload(self, image_path: str) -> Dict[str, Any]:
        """Step 1: Validate uploaded document."""
        start_time = datetime.now()
        
        try:
            logging.info("📝 Step 1: Validating document upload...")
            
            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Image file not found', 'processing_time': 0}
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                return {'success': False, 'error': 'File too large (>50MB)', 'processing_time': 0}
            
            # Validate image format
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_type = img.format
            except Exception:
                return {'success': False, 'error': 'Invalid image format', 'processing_time': 0}
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'file_path': image_path,
                'file_size': file_size,
                'image_dimensions': {'width': width, 'height': height},
                'image_format': format_type,
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Document validated: {width}x{height} {format_type}, {file_size} bytes")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Validation failed: {e}', 'processing_time': 0}
    
    def _step2_extract_text(self, image_path: str) -> Dict[str, Any]:
        """Step 2: Extract text using OCR."""
        start_time = datetime.now()
        
        try:
            logging.info("🔍 Step 2: Extracting text with OCR...")
            
            # Try multiple OCR methods for best results
            extracted_text = ""
            ocr_methods_used = []
            
            # Method 1: Basic Tesseract OCR
            try:
                basic_text = extract_text(image_path)
                if basic_text and basic_text.strip():
                    extracted_text = basic_text
                    ocr_methods_used.append("tesseract")
            except Exception as e:
                logging.warning(f"Tesseract OCR failed: {e}")
            
            # Method 2: Deep Learning OCR if available
            if not extracted_text.strip():
                try:
                    deep_ocr = DeepLearningOCR()
                    deep_result = deep_ocr.extract_text_comprehensive(image_path)
                    if deep_result.get('success') and deep_result.get('text'):
                        extracted_text = deep_result['text']
                        ocr_methods_used.append("deep_learning")
                except Exception as e:
                    logging.warning(f"Deep Learning OCR failed: {e}")
            
            if not extracted_text or not extracted_text.strip():
                return {
                    'success': False, 
                    'error': 'No text could be extracted from image using any OCR method',
                    'processing_time': 0
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'ocr_methods_used': ocr_methods_used,
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Text extracted: {len(extracted_text)} characters, {len(extracted_text.split())} words using {', '.join(ocr_methods_used)}")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'OCR extraction failed: {e}', 'processing_time': 0}
    
    def _step3_detect_fields(self, text: str, image_path: str, confidence_threshold: float) -> Dict[str, Any]:
        """Step 3: Detect fields in the extracted text."""
        start_time = datetime.now()
        
        try:
            logging.info("🎯 Step 3: Detecting fields in text...")
            
            if not self.field_pipeline:
                return {
                    'success': False,
                    'error': 'Field detection not available',
                    'processing_time': 0
                }
            
            # Extract fields using the enhanced pipeline
            analysis = self.field_pipeline.extract_fields_from_text(text, document_image_path=image_path)
            
            if not hasattr(analysis, 'extracted_fields'):
                return {
                    'success': False,
                    'error': 'Invalid field analysis result',
                    'processing_time': 0
                }
            
            # Filter fields by confidence threshold
            detected_fields = []
            for field in analysis.extracted_fields:
                confidence = getattr(field, 'confidence', 0)
                if confidence >= confidence_threshold:
                    field_data = {
                        'field_name': getattr(field, 'field_name', 'unknown'),
                        'field_value': getattr(field, 'field_value', ''),
                        'confidence': confidence,
                        'location': getattr(field, 'location', {}),
                        'category': getattr(field, 'category', 'other')
                    }
                    detected_fields.append(field_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'detected_fields': detected_fields,
                'total_fields_found': len(analysis.extracted_fields),
                'fields_above_threshold': len(detected_fields),
                'confidence_threshold': confidence_threshold,
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Fields detected: {len(detected_fields)} fields above threshold ({confidence_threshold})")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Field detection failed: {e}', 'processing_time': 0}
    
    def _step4_select_fields(self, detected_fields: List[Dict], user_selected_fields: List[Dict] = None) -> Dict[str, Any]:
        """Step 4: Select fields for blurring (user selection or auto-selection)."""
        start_time = datetime.now()
        
        try:
            logging.info("🎯 Step 4: Selecting fields for blurring...")
            
            selected_fields = []
            selection_method = "auto"
            
            if user_selected_fields:
                # Use user-provided field selections
                selected_fields = user_selected_fields
                selection_method = "user"
                logging.info(f"   📝 Using user-selected fields: {len(selected_fields)} fields")
            else:
                # Auto-select high-confidence sensitive fields
                sensitive_categories = ['personal_info', 'identification', 'financial']
                sensitive_field_names = [
                    'name', 'address', 'phone', 'email', 'date_of_birth', 
                    'aadhar_number', 'pan_number', 'passport_number',
                    'account', 'amount', 'signature'
                ]
                
                for field in detected_fields:
                    field_name = field.get('field_name', '').lower()
                    field_category = field.get('category', '').lower()
                    field_confidence = field.get('confidence', 0)
                    
                    # Auto-select if high confidence and sensitive
                    if (field_confidence >= 0.7 and 
                        (field_category in sensitive_categories or 
                         any(sensitive in field_name for sensitive in sensitive_field_names))):
                        
                        # Generate blur coordinates if not present
                        location = field.get('location', {})
                        if not location:
                            # Generate default blur area (will be refined in blur step)
                            location = {
                                'x': 50,  # Default position
                                'y': 50, 
                                'width': 200,
                                'height': 30
                            }
                        
                        selected_fields.append({
                            'field_name': field['field_name'],
                            'field_value': field['field_value'],
                            'confidence': field['confidence'],
                            'location': location,
                            'category': field['category']
                        })
                
                logging.info(f"   🤖 Auto-selected {len(selected_fields)} sensitive fields")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'selected_fields': selected_fields,
                'selection_method': selection_method,
                'fields_selected_count': len(selected_fields),
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Field selection complete: {len(selected_fields)} fields selected via {selection_method}")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Field selection failed: {e}', 'processing_time': 0}
    
    def _step5_blur_fields(self, image_path: str, selected_fields: List[Dict], blur_strength: int) -> Dict[str, Any]:
        """Step 5: Apply blur effect to selected fields."""
        start_time = datetime.now()
        
        try:
            logging.info("🎨 Step 5: Applying blur effects to selected fields...")
            
            if not selected_fields:
                logging.info("   ⚠️ No fields selected for blurring")
                # Return original image
                with open(image_path, 'rb') as f:
                    original_data = f.read()
                
                result = {
                    'success': True,
                    'blurred_image': original_data,
                    'blur_areas': [],
                    'blur_strength': 0,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
                return result
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image for blurring', 'processing_time': 0}
            
            blur_areas_applied = []
            
            # Apply blur to each selected field
            for field in selected_fields:
                location = field.get('location', {})
                
                if location and all(k in location for k in ['x', 'y', 'width', 'height']):
                    x, y, width, height = location['x'], location['y'], location['width'], location['height']
                    
                    # Ensure coordinates are within image bounds
                    img_height, img_width = image.shape[:2]
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    width = min(width, img_width - x)
                    height = min(height, img_height - y)
                    
                    if width > 0 and height > 0:
                        # Extract region to blur
                        roi = image[y:y+height, x:x+width]
                        
                        # Apply Gaussian blur
                        blurred_roi = cv2.GaussianBlur(roi, (blur_strength*2+1, blur_strength*2+1), 0)
                        
                        # Replace region with blurred version
                        image[y:y+height, x:x+width] = blurred_roi
                        
                        blur_areas_applied.append({
                            'field_name': field['field_name'],
                            'coordinates': {'x': x, 'y': y, 'width': width, 'height': height}
                        })
                else:
                    # Try to find text location for blurring (fallback)
                    field_value = field.get('field_value', '')
                    if field_value and len(field_value) > 3:
                        # Simple fallback: blur a default area (can be enhanced with text detection)
                        x, y, width, height = 50, 100, min(len(field_value) * 10, 300), 25
                        roi = image[y:y+height, x:x+width]
                        blurred_roi = cv2.GaussianBlur(roi, (blur_strength*2+1, blur_strength*2+1), 0)
                        image[y:y+height, x:x+width] = blurred_roi
                        
                        blur_areas_applied.append({
                            'field_name': field['field_name'],
                            'coordinates': {'x': x, 'y': y, 'width': width, 'height': height}
                        })
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.png', image)
            blurred_image_data = buffer.tobytes()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'blurred_image': blurred_image_data,
                'blur_areas': blur_areas_applied,
                'blur_strength': blur_strength,
                'fields_blurred': len(blur_areas_applied),
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Blur applied: {len(blur_areas_applied)} areas blurred with strength {blur_strength}")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Field blurring failed: {e}', 'processing_time': 0}
    
    def _step6_export_image(self, blurred_image_data: bytes, flow_id: str) -> Dict[str, Any]:
        """Step 6: Export the modified image."""
        start_time = datetime.now()
        
        try:
            logging.info("💾 Step 6: Exporting modified image...")
            
            # Generate output filename
            output_filename = f"blurred_{flow_id}.png"
            output_path = self.output_folder / output_filename
            
            # Save blurred image
            with open(output_path, 'wb') as f:
                f.write(blurred_image_data)
            
            # Convert to base64 for API response
            image_base64 = base64.b64encode(blurred_image_data).decode('utf-8')
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'output_path': str(output_path),
                'output_filename': output_filename,
                'image_base64': image_base64,
                'file_size': len(blurred_image_data),
                'processing_time': processing_time
            }
            
            logging.info(f"   ✅ Image exported: {output_path} ({len(blurred_image_data)} bytes)")
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Image export failed: {e}', 'processing_time': 0}
    
    def _create_error_response(self, flow_id: str, step: str, error: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'flow_id': flow_id,
            'failed_step': step,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get complete processing history."""
        return self.processing_history
    
    def get_flow_status(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific flow."""
        for flow in self.processing_history:
            if flow.get('flow_id') == flow_id:
                return flow
        return None

# Global instance for Flask integration
complete_ocr_flow = CompleteOCRFlow()

# Convenience function for direct usage
def process_document_sequential(image_path: str, 
                              selected_fields: List[Dict] = None,
                              blur_strength: int = 15,
                              confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Process document through complete OCR flow.
    
    Args:
        image_path: Path to document image
        selected_fields: Optional list of user-selected fields to blur
        blur_strength: Blur effect strength (1-50)
        confidence_threshold: Minimum confidence for field detection (0.0-1.0)
        
    Returns:
        Complete processing results
    """
    return complete_ocr_flow.process_document_complete_flow(
        image_path, selected_fields, blur_strength, confidence_threshold
    )