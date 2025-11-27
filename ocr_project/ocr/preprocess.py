"""
Advanced Image Preprocessing Module for OCR Enhancement
Comprehensive OpenCV and Pillow-based preprocessing strategies
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import restoration, filters, morphology, segmentation
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedImagePreprocessor:
    """Advanced image preprocessing class with multiple strategies for OCR enhancement."""
    
    def __init__(self):
        """Initialize the preprocessor with available strategies."""
        self.strategies = {
            'grayscale_enhanced': self.grayscale_enhanced,
            'high_contrast': self.high_contrast,
            'adaptive_threshold': self.adaptive_threshold,
            'noise_removal': self.noise_removal,
            'deskewing': self.deskewing,
            'morphological': self.morphological,
            'color_quantization': self.color_quantization,
            'text_enhancement': self.text_enhancement,
            'shadow_removal': self.shadow_removal
        }
        logger.info(f"✅ AdvancedImagePreprocessor initialized with {len(self.strategies)} strategies")
    
    def apply_strategy(self, image: np.ndarray, strategy_name: str) -> np.ndarray:
        """Apply a specific preprocessing strategy to an image."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        try:
            return self.strategies[strategy_name](image)
        except Exception as e:
            logger.error(f"Failed to apply strategy {strategy_name}: {e}")
            raise
    
    def grayscale_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale with enhanced contrast and sharpening."""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            # Use weighted grayscale conversion for better text visibility
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(equalized)
        
        # Apply unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 2.0)
        unsharp_mask = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    def high_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply high contrast enhancement with CLAHE."""
        # Convert to LAB color space for better contrast control
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        if len(image.shape) == 3:
            # Merge channels and convert back
            enhanced_lab = cv2.merge([enhanced_l, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            result = enhanced_l
        
        # Additional contrast enhancement using Pillow
        if len(result.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(result)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        contrasted = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Brightness(contrasted)
        brightened = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        final_result = np.array(brightened)
        if len(final_result.shape) == 3 and len(image.shape) == 3:
            final_result = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
        
        return final_result
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for varying lighting conditions."""
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive
    
    def noise_removal(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text details."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Apply median filter to remove salt and pepper noise
        median = cv2.medianBlur(bilateral, 3)
        
        return median
    
    def deskewing(self, image: np.ndarray) -> np.ndarray:
        """Correct skewed text using Hough line detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    skew_angle = np.median(angles)
                    
                    # Rotate image to correct skew
                    (h, w) = gray.shape
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    
                    # Calculate new dimensions to avoid cropping
                    cos = np.abs(rotation_matrix[0, 0])
                    sin = np.abs(rotation_matrix[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    
                    # Adjust rotation matrix for new dimensions
                    rotation_matrix[0, 2] += (new_w / 2) - center[0]
                    rotation_matrix[1, 2] += (new_h / 2) - center[1]
                    
                    deskewed = cv2.warpAffine(gray, rotation_matrix, (new_w, new_h), 
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    return deskewed
            
            return gray
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    def morphological(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance text structure."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        
        # Opening (erosion followed by dilation) to remove noise
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion) to fill gaps in text
        kernel2 = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
        
        return closed
    
    def color_quantization(self, image: np.ndarray) -> np.ndarray:
        """Reduce color complexity using K-means clustering."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering to reduce colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 4  # Number of color clusters
        
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape to original image shape
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(image.shape)
        
        return quantized_image
    
    def text_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Specialized text enhancement using morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to strengthen text strokes
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create enhanced text by combining original with edge information
        enhanced = cv2.addWeighted(gray, 0.8, dilated, 0.2, 0)
        
        # Apply sharpening filter
        kernel_sharp = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def shadow_removal(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows and uneven illumination."""
        if len(image.shape) == 3:
            # Convert to grayscale for shadow detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Estimate background using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Remove background by subtraction
        normalized = cv2.subtract(background, gray)
        normalized = cv2.add(normalized, 100)  # Adjust brightness
        
        # Apply histogram equalization for uniform lighting
        equalized = cv2.equalizeHist(normalized)
        
        return equalized
    
    def document_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Optimized pipeline for general documents."""
        try:
            # Step 1: Convert to enhanced grayscale
            processed = self.grayscale_enhanced(image)
            
            # Step 2: Remove noise
            processed = self.noise_removal(processed)
            
            # Step 3: Remove shadows
            processed = self.shadow_removal(processed)
            
            # Step 4: Apply adaptive thresholding
            processed = self.adaptive_threshold(processed)
            
            # Step 5: Correct skewing
            processed = self.deskewing(processed)
            
            # Step 6: Enhance text
            processed = self.text_enhancement(processed)
            
            return processed
        except Exception as e:
            logger.error(f"Document pipeline failed: {e}")
            return image
    
    def id_card_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Specialized pipeline for ID cards and official documents."""
        try:
            # Step 1: Apply high contrast enhancement
            processed = self.high_contrast(image)
            
            # Step 2: Remove shadows (critical for laminated cards)
            processed = self.shadow_removal(processed)
            
            # Step 3: Color quantization to separate text from background/logos
            processed = self.color_quantization(processed)
            
            # Step 4: Remove noise
            processed = self.noise_removal(processed)
            
            # Step 5: Enhance text structure
            processed = self.text_enhancement(processed)
            
            # Step 6: Apply morphological operations
            processed = self.morphological(processed)
            
            return processed
        except Exception as e:
            logger.error(f"ID card pipeline failed: {e}")
            return image
    
    def handwritten_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Optimized pipeline for handwritten text."""
        try:
            # Step 1: Convert to enhanced grayscale
            processed = self.grayscale_enhanced(image)
            
            # Step 2: Apply high contrast
            processed = self.high_contrast(processed)
            
            # Step 3: Adaptive thresholding for ink detection
            processed = self.adaptive_threshold(processed)
            
            # Step 4: Remove noise carefully (preserve pen strokes)
            processed = self.noise_removal(processed)
            
            # Step 5: Enhance text with larger kernels for handwriting
            processed = self.text_enhancement(processed)
            
            # Step 6: Correct any skewing
            processed = self.deskewing(processed)
            
            return processed
        except Exception as e:
            logger.error(f"Handwritten pipeline failed: {e}")
            return image


def preprocess_with_advanced_opencv(image_path: str, preprocessing_strategy: str = "auto") -> Dict[str, Any]:
    """
    Advanced OpenCV-based preprocessing with comprehensive strategies.
    
    Args:
        image_path: Path to the image file
        preprocessing_strategy: Strategy to use ('auto', 'document', 'id_card', 'handwritten', or specific strategy name)
    
    Returns:
        Dict containing processed image and metadata
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Initialize preprocessor
        preprocessor = AdvancedImagePreprocessor()
        
        # Apply preprocessing based on strategy
        if preprocessing_strategy == "auto" or preprocessing_strategy == "document":
            processed_image = preprocessor.document_pipeline(image)
        elif preprocessing_strategy == "id_card":
            processed_image = preprocessor.id_card_pipeline(image)
        elif preprocessing_strategy == "handwritten":
            processed_image = preprocessor.handwritten_pipeline(image)
        else:
            # Try to apply specific strategy
            processed_image = preprocessor.apply_strategy(image, preprocessing_strategy)
        
        return {
            'success': True,
            'processed_image': processed_image,
            'original_shape': image.shape,
            'processed_shape': processed_image.shape,
            'strategy': preprocessing_strategy
        }
        
    except Exception as e:
        logger.error(f"Advanced preprocessing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'strategy': preprocessing_strategy
        }


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format."""
    if len(cv_image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(cv_image)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Additional legacy functions for backward compatibility
def load_image(image_path: str) -> Image.Image:
    """Legacy function - loads image as PIL Image."""
    from PIL import Image
    return Image.open(image_path)

def to_grayscale(image):
    """Convert image to grayscale."""
    if isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    else:
        return image.convert('L')

def apply_threshold(image, threshold=127):
    """Apply binary threshold to image."""
    if isinstance(image, np.ndarray):
        _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    else:
        return image.point(lambda x: 255 if x > threshold else 0, mode='1')

def denoise_image(image):
    """Remove noise from image."""
    if isinstance(image, np.ndarray):
        return cv2.medianBlur(image, 3)
    else:
        return image.filter(ImageFilter.MedianFilter(size=3))

def enhance_contrast(image):
    """Enhance image contrast."""
    if isinstance(image, np.ndarray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    else:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)

def resize_image(image, scale_factor=2.0):
    """Resize image by scale factor."""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    else:
        width, height = image.size
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)