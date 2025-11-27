"""
Deep Learning OCR Module
Advanced OCR using EasyOCR and PaddleOCR neural networks with advanced preprocessing
Supports multiple languages and provides superior accuracy through image enhancement
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from .preprocess import preprocess_with_advanced_opencv, AdvancedImagePreprocessor
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepLearningOCR:
    """Advanced OCR using deep learning models with comprehensive preprocessing."""
    
    def __init__(self):
        """Initialize deep learning OCR engines and advanced preprocessor."""
        self.easyocr_reader = None
        self.paddleocr_reader = None
        self.torch_available = False
        self.paddle_available = False
        
        # Initialize advanced image preprocessor
        self.preprocessor = AdvancedImagePreprocessor()
        logger.info("✅ Advanced Image Preprocessor initialized")
        
        # Initialize engines
        self._initialize_easyocr()
        self._initialize_paddleocr()
        
        # Default configuration
        self.default_config = {
            'confidence_threshold': 0.6,
            'languages': ['en'],
            'use_gpu': False,
            'paragraph_mode': False,
            'width_threshold': 0.7,
            'height_threshold': 0.7
        }
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR with error handling."""
        try:
            import easyocr
            # Initialize with English by default, can be extended
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR initialized successfully")
            return True
        except ImportError:
            logger.warning("⚠️ EasyOCR not installed. Install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            return False
    
    def _initialize_paddleocr(self):
        """Initialize PaddleOCR with error handling."""
        try:
            from paddleocr import PaddleOCR
            # Initialize with English, can be extended to other languages
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.paddle_available = True
            logger.info("✅ PaddleOCR initialized successfully")
            return True
        except ImportError:
            logger.warning("⚠️ PaddleOCR not installed. Install with: pip install paddleocr")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize PaddleOCR: {e}")
            return False
    
    def extract_text_easyocr(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Extract text using EasyOCR with advanced preprocessing strategies."""
        if not self.easyocr_reader:
            return {'error': 'EasyOCR not available', 'text': '', 'confidence': 0.0}
        
        try:
            start_time = time.time()
            
            # Read original image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image', 'text': '', 'confidence': 0.0}
            
            # Use advanced preprocessing strategies for optimal text detection
            preprocessing_strategies = []
            
            # Original image
            preprocessing_strategies.append({'name': 'original', 'image': image})
            
            # Try all advanced preprocessing strategies from AdvancedImagePreprocessor
            for strategy_name in ['grayscale_enhanced', 'high_contrast', 'adaptive_threshold', 
                                'noise_removal', 'deskewing', 'morphological', 'color_quantization', 
                                'text_enhancement', 'shadow_removal']:
                try:
                    preprocessed_img = self.preprocessor.apply_strategy(image, strategy_name)
                    preprocessing_strategies.append({
                        'name': strategy_name, 
                        'image': preprocessed_img
                    })
                except Exception as e:
                    logger.warning(f"Failed to apply preprocessing strategy '{strategy_name}': {e}")
                    continue
            
            # Try document-specific pipelines
            for pipeline_name in ['document_pipeline', 'id_card_pipeline', 'handwritten_pipeline']:
                try:
                    if pipeline_name == 'document_pipeline':
                        preprocessed_img = self.preprocessor.document_pipeline(image)
                    elif pipeline_name == 'id_card_pipeline':
                        preprocessed_img = self.preprocessor.id_card_pipeline(image)
                    else:
                        preprocessed_img = self.preprocessor.handwritten_pipeline(image)
                    
                    preprocessing_strategies.append({
                        'name': pipeline_name, 
                        'image': preprocessed_img
                    })
                except Exception as e:
                    logger.warning(f"Failed to apply preprocessing pipeline '{pipeline_name}': {e}")
                    continue
            
            best_result = None
            best_score = 0
            strategy_results = []
            
            for strategy in preprocessing_strategies:
                try:
                    # Use optimized EasyOCR parameters for better detection
                    results = self.easyocr_reader.readtext(
                        strategy['image'], 
                        paragraph=False,
                        width_ths=0.3,    # Lower width threshold for better text detection
                        height_ths=0.3,   # Lower height threshold
                        mag_ratio=1.8,    # Higher magnification for small text
                        slope_ths=0.3,    # Allow more slope tolerance
                        ycenter_ths=0.7,  # Y-center threshold for line detection
                        low_text=0.2      # Lower text confidence threshold
                    )
                    
                    # Process results with adaptive confidence threshold
                    extracted_data = self._process_easyocr_results(
                        results, 
                        confidence_threshold=0.2,  # Very low threshold for difficult text
                        **kwargs
                    )
                    
                    # Calculate comprehensive score
                    text_length = len(extracted_data.get('text', ''))
                    avg_confidence = extracted_data.get('confidence', 0)
                    word_count = len(extracted_data.get('text', '').split())
                    
                    # Weighted scoring: prioritize text length and word count
                    score = (text_length * 0.4 + 
                            avg_confidence * 0.3 + 
                            word_count * 0.3)
                    
                    strategy_result = {
                        'strategy': strategy['name'],
                        'text_length': text_length,
                        'confidence': avg_confidence,
                        'word_count': word_count,
                        'score': score,
                        'text_preview': extracted_data.get('text', '')[:50] + ('...' if len(extracted_data.get('text', '')) > 50 else '')
                    }
                    strategy_results.append(strategy_result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = extracted_data
                        best_result['preprocessing_strategy'] = strategy['name']
                        best_result['strategy_score'] = score
                    
                    logger.info(f"EasyOCR strategy '{strategy['name']}': {text_length} chars, {word_count} words, conf: {avg_confidence:.3f}, score: {score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"EasyOCR strategy '{strategy['name']}' failed: {e}")
                    continue
            
            if best_result is None:
                return {'error': 'All EasyOCR preprocessing strategies failed', 'text': '', 'confidence': 0.0}
            
            processing_time = time.time() - start_time
            best_result['processing_time'] = processing_time
            best_result['engine'] = 'EasyOCR'
            best_result['strategy_results'] = strategy_results  # Include all strategy results for debugging
            
            logger.info(f"EasyOCR processing completed in {processing_time:.2f}s using '{best_result.get('preprocessing_strategy', 'unknown')}' strategy with score {best_result.get('strategy_score', 0):.3f}")
            return best_result
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'error': str(e), 'text': '', 'confidence': 0.0}
    
    def extract_text_paddleocr(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        if not self.paddleocr_reader:
            return {'error': 'PaddleOCR not available', 'text': '', 'confidence': 0.0}
        
        try:
            start_time = time.time()
            
            # PaddleOCR processing
            results = self.paddleocr_reader.ocr(image_path, cls=True)
            
            # Process results
            extracted_data = self._process_paddleocr_results(results, **kwargs)
            
            processing_time = time.time() - start_time
            extracted_data['processing_time'] = processing_time
            extracted_data['engine'] = 'PaddleOCR'
            
            logger.info(f"PaddleOCR processing completed in {processing_time:.2f}s")
            return extracted_data
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {'error': str(e), 'text': '', 'confidence': 0.0}
    
    def extract_text_hybrid(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Extract text using both engines and combine results."""
        results = {}
        
        # Try EasyOCR
        if self.easyocr_reader:
            easy_result = self.extract_text_easyocr(image_path, **kwargs)
            results['easyocr'] = easy_result
        
        # Try PaddleOCR
        if self.paddleocr_reader:
            paddle_result = self.extract_text_paddleocr(image_path, **kwargs)
            results['paddleocr'] = paddle_result
        
        # Combine and select best result
        return self._combine_results(results, **kwargs)
    
    def _process_easyocr_results(self, results: List, **kwargs) -> Dict[str, Any]:
        """Process EasyOCR results into structured format with improved handling."""
        confidence_threshold = kwargs.get('confidence_threshold', 0.3)  # Lower default threshold
        
        text_blocks = []
        all_text = []
        total_confidence = 0
        valid_detections = 0
        bounding_boxes = []
        
        # Sort results by confidence to prioritize high-confidence detections
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        for detection in sorted_results:
            bbox, text, confidence = detection
            
            # Clean and validate text
            cleaned_text = text.strip()
            if not cleaned_text:
                continue
                
            # Be more lenient with confidence for longer text (likely to be names/important text)
            adjusted_threshold = confidence_threshold
            if len(cleaned_text) > 8:  # For longer text like names
                adjusted_threshold = max(0.2, confidence_threshold - 0.2)
            
            if confidence >= adjusted_threshold:
                text_blocks.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'area': self._calculate_bbox_area(bbox)
                })
                all_text.append(cleaned_text)
                total_confidence += confidence
                valid_detections += 1
                bounding_boxes.append(bbox)
                
                logger.debug(f"EasyOCR detected: '{cleaned_text}' (conf: {confidence:.3f})")
        
        # Calculate overall confidence
        avg_confidence = total_confidence / valid_detections if valid_detections > 0 else 0
        
        # Combine text with better spacing
        combined_text = ' '.join(all_text) if all_text else ''
        
        return {
            'text': combined_text,
            'confidence': avg_confidence,
            'text_blocks': text_blocks,
            'total_detections': len(results),
            'valid_detections': valid_detections,
            'bounding_boxes': bounding_boxes,
            'metadata': {
                'engine': 'EasyOCR',
                'confidence_threshold': confidence_threshold,
                'language': 'en',
                'adjusted_threshold_used': True
            }
        }
    
    def _process_paddleocr_results(self, results: List, **kwargs) -> Dict[str, Any]:
        """Process PaddleOCR results into structured format."""
        confidence_threshold = kwargs.get('confidence_threshold', self.default_config['confidence_threshold'])
        
        text_blocks = []
        all_text = []
        total_confidence = 0
        valid_detections = 0
        bounding_boxes = []
        
        if results and results[0]:
            for detection in results[0]:
                bbox, (text, confidence) = detection
                
                if confidence >= confidence_threshold:
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'area': self._calculate_bbox_area(bbox)
                    })
                    all_text.append(text)
                    total_confidence += confidence
                    valid_detections += 1
                    bounding_boxes.append(bbox)
        
        # Calculate overall confidence
        avg_confidence = total_confidence / valid_detections if valid_detections > 0 else 0
        
        # Combine text
        combined_text = ' '.join(all_text) if all_text else ''
        
        return {
            'text': combined_text,
            'confidence': avg_confidence,
            'text_blocks': text_blocks,
            'total_detections': len(results[0]) if results and results[0] else 0,
            'valid_detections': valid_detections,
            'bounding_boxes': bounding_boxes,
            'metadata': {
                'engine': 'PaddleOCR',
                'confidence_threshold': confidence_threshold,
                'language': 'en'
            }
        }
    
    def _combine_results(self, results: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
        """Combine results from multiple engines and select the best."""
        if not results:
            return {'error': 'No OCR engines available', 'text': '', 'confidence': 0.0}
        
        best_result = None
        best_score = 0
        
        for engine_name, result in results.items():
            if 'error' not in result:
                # Score based on confidence and text length
                score = result.get('confidence', 0) * 0.7 + min(len(result.get('text', '')), 1000) / 1000 * 0.3
                if score > best_score:
                    best_score = score
                    best_result = result
        
        if best_result:
            best_result['hybrid_results'] = results
            best_result['selection_score'] = best_score
            return best_result
        else:
            # Return first available result if no good one found
            return list(results.values())[0]
    
    def _calculate_bbox_area(self, bbox: List) -> float:
        """Calculate area of bounding box."""
        try:
            if len(bbox) == 4 and len(bbox[0]) == 2:
                # Convert to numpy array for easier calculation
                points = np.array(bbox)
                # Calculate area using shoelace formula
                x = points[:, 0]
                y = points[:, 1]
                area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
                return area
            return 0
        except Exception:
            return 0
    
    def _enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR results on ID cards."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold to enhance text
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Apply slight blur to smooth edges
            blurred = cv2.GaussianBlur(cleaned, (1, 1), 0)
            
            return blurred
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _apply_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply high contrast enhancement for better text visibility."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply additional contrast enhancement
            contrast_enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=10)
            
            return contrast_enhanced
            
        except Exception as e:
            logger.warning(f"High contrast enhancement failed: {e}")
            return image
    
    def extract_text_with_layout(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Extract text with advanced layout analysis."""
        # Use the best available engine
        if self.paddleocr_reader:
            result = self.extract_text_paddleocr(image_path, **kwargs)
        elif self.easyocr_reader:
            result = self.extract_text_easyocr(image_path, **kwargs)
        else:
            return {'error': 'No deep learning OCR engines available', 'text': '', 'confidence': 0.0}
        
        # Add layout analysis
        if 'text_blocks' in result:
            result['layout_analysis'] = self._analyze_layout(result['text_blocks'])
        
        return result
    
    def _analyze_layout(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Analyze document layout from text blocks."""
        if not text_blocks:
            return {'structure': 'unknown', 'regions': []}
        
        # Sort blocks by position (top to bottom, left to right)
        sorted_blocks = sorted(text_blocks, key=lambda b: (
            min([point[1] for point in b['bbox']]),  # Top position
            min([point[0] for point in b['bbox']])   # Left position
        ))
        
        # Analyze structure
        layout = {
            'structure': self._detect_document_structure(sorted_blocks),
            'regions': self._identify_regions(sorted_blocks),
            'reading_order': [block['text'] for block in sorted_blocks],
            'total_blocks': len(sorted_blocks),
            'confidence_distribution': self._analyze_confidence_distribution(sorted_blocks)
        }
        
        return layout
    
    def _detect_document_structure(self, blocks: List[Dict]) -> str:
        """Detect document structure type."""
        if len(blocks) == 0:
            return 'empty'
        elif len(blocks) <= 3:
            return 'simple'
        elif any('invoice' in block['text'].lower() or 'bill' in block['text'].lower() for block in blocks):
            return 'invoice'
        elif any('receipt' in block['text'].lower() or 'total' in block['text'].lower() for block in blocks):
            return 'receipt'
        else:
            return 'document'
    
    def _identify_regions(self, blocks: List[Dict]) -> List[Dict]:
        """Identify different regions in the document."""
        regions = []
        
        if not blocks:
            return regions
        
        # Header region (top 20%)
        image_height = max([max([point[1] for point in block['bbox']]) for block in blocks])
        header_threshold = image_height * 0.2
        
        header_blocks = [b for b in blocks if min([point[1] for point in b['bbox']]) <= header_threshold]
        if header_blocks:
            regions.append({
                'type': 'header',
                'blocks': len(header_blocks),
                'text': ' '.join([b['text'] for b in header_blocks])
            })
        
        # Body region (middle 60%)
        body_blocks = [b for b in blocks if header_threshold < min([point[1] for point in b['bbox']]) <= image_height * 0.8]
        if body_blocks:
            regions.append({
                'type': 'body',
                'blocks': len(body_blocks),
                'text': ' '.join([b['text'] for b in body_blocks])
            })
        
        # Footer region (bottom 20%)
        footer_blocks = [b for b in blocks if min([point[1] for point in b['bbox']]) > image_height * 0.8]
        if footer_blocks:
            regions.append({
                'type': 'footer',
                'blocks': len(footer_blocks),
                'text': ' '.join([b['text'] for b in footer_blocks])
            })
        
        return regions
    
    def _analyze_confidence_distribution(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Analyze confidence distribution of text blocks."""
        if not blocks:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        confidences = [block['confidence'] for block in blocks]
        
        return {
            'mean': np.mean(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'std': np.std(confidences),
            'high_confidence_blocks': len([c for c in confidences if c > 0.8]),
            'low_confidence_blocks': len([c for c in confidences if c < 0.5])
        }
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        engines = []
        if self.easyocr_reader:
            engines.append('EasyOCR')
        if self.paddleocr_reader:
            engines.append('PaddleOCR')
        return engines
    
    def benchmark_engines(self, image_path: str) -> Dict[str, Any]:
        """Benchmark different engines on the same image."""
        benchmark_results = {}
        
        if self.easyocr_reader:
            benchmark_results['EasyOCR'] = self.extract_text_easyocr(image_path)
        
        if self.paddleocr_reader:
            benchmark_results['PaddleOCR'] = self.extract_text_paddleocr(image_path)
        
        # Add comparison metrics
        if len(benchmark_results) > 1:
            benchmark_results['comparison'] = self._compare_results(benchmark_results)
        
        return benchmark_results
    
    def _compare_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare results from different engines."""
        comparison = {
            'engines': list(results.keys()),
            'text_lengths': {},
            'confidences': {},
            'processing_times': {},
            'agreement_score': 0
        }
        
        texts = []
        for engine, result in results.items():
            if 'error' not in result:
                comparison['text_lengths'][engine] = len(result.get('text', ''))
                comparison['confidences'][engine] = result.get('confidence', 0)
                comparison['processing_times'][engine] = result.get('processing_time', 0)
                texts.append(result.get('text', ''))
        
        # Calculate text similarity (simple word overlap)
        if len(texts) == 2:
            words1 = set(texts[0].lower().split())
            words2 = set(texts[1].lower().split())
            if words1 or words2:
                comparison['agreement_score'] = len(words1.intersection(words2)) / len(words1.union(words2))
        
        return comparison


# Global instance
dl_ocr = DeepLearningOCR()


def extract_text_deep_learning(image_path: str, engine: str = 'auto', **kwargs) -> Dict[str, Any]:
    """
    Main function to extract text using deep learning OCR.
    
    Args:
        image_path: Path to the image file
        engine: OCR engine to use ('easyocr', 'paddleocr', 'hybrid', 'auto')
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        if engine == 'easyocr':
            return dl_ocr.extract_text_easyocr(image_path, **kwargs)
        elif engine == 'paddleocr':
            return dl_ocr.extract_text_paddleocr(image_path, **kwargs)
        elif engine == 'hybrid':
            return dl_ocr.extract_text_hybrid(image_path, **kwargs)
        else:  # auto
            # Use the best available engine
            available_engines = dl_ocr.get_available_engines()
            if 'PaddleOCR' in available_engines:
                return dl_ocr.extract_text_paddleocr(image_path, **kwargs)
            elif 'EasyOCR' in available_engines:
                return dl_ocr.extract_text_easyocr(image_path, **kwargs)
            else:
                return {'error': 'No deep learning OCR engines available', 'text': '', 'confidence': 0.0}
    
    except Exception as e:
        logger.error(f"Deep learning OCR failed: {e}")
        return {'error': str(e), 'text': '', 'confidence': 0.0}


def get_deep_learning_ocr_info() -> Dict[str, Any]:
    """Get information about available deep learning OCR engines."""
    return {
        'available_engines': dl_ocr.get_available_engines(),
        'easyocr_available': dl_ocr.easyocr_reader is not None,
        'paddleocr_available': dl_ocr.paddleocr_reader is not None,
        'default_config': dl_ocr.default_config
    }


if __name__ == "__main__":
    # Test the deep learning OCR
    print("Testing Deep Learning OCR...")
    info = get_deep_learning_ocr_info()
    print(f"Available engines: {info['available_engines']}")
    
    # Test with a sample image if available
    test_image = "test_image.png"
    if os.path.exists(test_image):
        result = extract_text_deep_learning(test_image)
        print(f"Test result: {result}")