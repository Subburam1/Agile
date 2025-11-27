"""OCR Image to Text Extraction Package"""

from .ocr import extract_text, extract_text_from_file
from .preprocess import preprocess_with_advanced_opencv, AdvancedImagePreprocessor
from .certificate_ocr import extract_certificate_text, preprocess_certificate, enhance_certificate_image_quality
from .mrz_ocr import extract_mrz_text, detect_mrz_region, preprocess_mrz_image, MRZParser, extract_raw_mrz_text
from .document_types import DocumentTypeDetector, DocumentProcessor
from .deep_learning_ocr import DeepLearningOCR
from .document_processor import DocumentProcessor as AdvancedDocumentProcessor, process_document_file, get_supported_formats

# Legacy functions for backward compatibility
def preprocess_image(image_path):
    """Legacy function - use preprocess_with_advanced_opencv instead."""
    result = preprocess_with_advanced_opencv(image_path)
    return result.get('processed_image', None) if result.get('success') else None

def load_image(image_path):
    """Legacy function for loading images."""
    import cv2
    return cv2.imread(image_path)

def preprocess_for_text_type(image, text_type="document"):
    """Legacy function - use AdvancedImagePreprocessor pipelines instead."""
    preprocessor = AdvancedImagePreprocessor()
    if text_type == "id_card":
        return preprocessor.id_card_pipeline(image)
    elif text_type == "handwritten":
        return preprocessor.handwritten_pipeline(image)
    else:
        return preprocessor.document_pipeline(image)

__version__ = "1.0.0"
__all__ = [
    "extract_text", "extract_text_from_file", 
    "preprocess_with_advanced_opencv", "AdvancedImagePreprocessor",
    "extract_certificate_text", "preprocess_certificate", "enhance_certificate_image_quality",
    "extract_mrz_text", "detect_mrz_region", "preprocess_mrz_image", "MRZParser", "extract_raw_mrz_text",
    "DocumentTypeDetector", "DocumentProcessor", "DeepLearningOCR",
    "AdvancedDocumentProcessor", "process_document_file", "get_supported_formats",
    # Legacy functions
    "preprocess_image", "load_image", "preprocess_for_text_type"
]