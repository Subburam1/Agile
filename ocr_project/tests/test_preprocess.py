"""Tests for image preprocessing functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile

from ocr.preprocess import (
    load_image, pil_to_cv2, cv2_to_pil, to_grayscale, 
    apply_threshold, denoise_image, enhance_contrast, 
    resize_image, preprocess_image, preprocess_for_text_type
)


class TestImageLoading:
    
    @patch('ocr.preprocess.Image.open')
    @patch('ocr.preprocess.Path.exists')
    def test_load_image_success(self, mock_exists, mock_image_open):
        """Test successful image loading."""
        # Arrange
        mock_exists.return_value = True
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image
        
        # Act
        result = load_image("test.jpg")
        
        # Assert
        assert result == mock_image
        mock_image_open.assert_called_once()
    
    @patch('ocr.preprocess.Path.exists')
    def test_load_image_file_not_found(self, mock_exists):
        """Test FileNotFoundError when image doesn't exist."""
        # Arrange
        mock_exists.return_value = False
        
        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            load_image("nonexistent.jpg")
        assert "Image file not found" in str(exc_info.value)
    
    @patch('ocr.preprocess.Image.open')
    @patch('ocr.preprocess.Path.exists')
    def test_load_image_open_error(self, mock_exists, mock_image_open):
        """Test exception handling during image opening."""
        # Arrange
        mock_exists.return_value = True
        mock_image_open.side_effect = Exception("Corrupted file")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            load_image("corrupted.jpg")
        assert "Failed to load image" in str(exc_info.value)


class TestImageConversion:
    
    @patch('ocr.preprocess.cv2.cvtColor')
    @patch('ocr.preprocess.np.array')
    def test_pil_to_cv2(self, mock_array, mock_cvtColor):
        """Test PIL to OpenCV conversion."""
        # Arrange
        mock_image = Mock(spec=Image.Image)
        mock_array.return_value = np.array([[1, 2, 3]])
        mock_cvtColor.return_value = np.array([[3, 2, 1]])
        
        # Act
        result = pil_to_cv2(mock_image)
        
        # Assert
        mock_array.assert_called_once_with(mock_image)
        mock_cvtColor.assert_called_once()
    
    @patch('ocr.preprocess.Image.fromarray')
    @patch('ocr.preprocess.cv2.cvtColor')
    def test_cv2_to_pil(self, mock_cvtColor, mock_fromarray):
        """Test OpenCV to PIL conversion."""
        # Arrange
        mock_cv_image = np.array([[1, 2, 3]])
        mock_converted = np.array([[3, 2, 1]])
        mock_cvtColor.return_value = mock_converted
        mock_pil_image = Mock(spec=Image.Image)
        mock_fromarray.return_value = mock_pil_image
        
        # Act
        result = cv2_to_pil(mock_cv_image)
        
        # Assert
        assert result == mock_pil_image
        mock_cvtColor.assert_called_once()
        mock_fromarray.assert_called_once_with(mock_converted)


class TestImageProcessing:
    
    def test_to_grayscale_with_pil_image(self):
        """Test grayscale conversion with PIL image."""
        with patch('ocr.preprocess.pil_to_cv2') as mock_pil_to_cv2, \
             patch('ocr.preprocess.cv2.cvtColor') as mock_cvtColor:
            
            # Arrange
            mock_image = Mock(spec=Image.Image)
            mock_cv_image = np.ones((100, 100, 3))
            mock_gray = np.ones((100, 100))
            mock_pil_to_cv2.return_value = mock_cv_image
            mock_cvtColor.return_value = mock_gray
            
            # Act
            result = to_grayscale(mock_image)
            
            # Assert
            mock_pil_to_cv2.assert_called_once_with(mock_image)
            mock_cvtColor.assert_called_once()
    
    def test_to_grayscale_with_cv2_color_image(self):
        """Test grayscale conversion with color OpenCV image."""
        with patch('ocr.preprocess.cv2.cvtColor') as mock_cvtColor:
            # Arrange
            color_image = np.ones((100, 100, 3))
            mock_gray = np.ones((100, 100))
            mock_cvtColor.return_value = mock_gray
            
            # Act
            result = to_grayscale(color_image)
            
            # Assert
            mock_cvtColor.assert_called_once()
    
    def test_to_grayscale_with_cv2_gray_image(self):
        """Test grayscale conversion with already gray OpenCV image."""
        # Arrange
        gray_image = np.ones((100, 100))
        
        # Act
        result = to_grayscale(gray_image)
        
        # Assert
        assert np.array_equal(result, gray_image)
    
    @patch('ocr.preprocess.cv2.threshold')
    def test_apply_threshold_otsu(self, mock_threshold):
        """Test Otsu thresholding."""
        # Arrange
        image = np.ones((100, 100))
        mock_threshold.return_value = (127, np.ones((100, 100)))
        
        # Act
        result = apply_threshold(image, method='otsu')
        
        # Assert
        mock_threshold.assert_called_once()
    
    @patch('ocr.preprocess.cv2.threshold')
    def test_apply_threshold_binary(self, mock_threshold):
        """Test binary thresholding."""
        # Arrange
        image = np.ones((100, 100))
        mock_threshold.return_value = (127, np.ones((100, 100)))
        
        # Act
        result = apply_threshold(image, method='binary', threshold_value=100)
        
        # Assert
        mock_threshold.assert_called_once()
    
    @patch('ocr.preprocess.cv2.adaptiveThreshold')
    def test_apply_threshold_adaptive(self, mock_adaptive):
        """Test adaptive thresholding."""
        # Arrange
        image = np.ones((100, 100))
        mock_adaptive.return_value = np.ones((100, 100))
        
        # Act
        result = apply_threshold(image, method='adaptive')
        
        # Assert
        mock_adaptive.assert_called_once()
    
    def test_apply_threshold_unknown_method(self):
        """Test error handling for unknown threshold method."""
        # Arrange
        image = np.ones((100, 100))
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            apply_threshold(image, method='unknown')
        assert "Unknown threshold method" in str(exc_info.value)
    
    @patch('ocr.preprocess.cv2.bilateralFilter')
    def test_denoise_bilateral(self, mock_bilateral):
        """Test bilateral denoising."""
        # Arrange
        image = np.ones((100, 100))
        mock_bilateral.return_value = np.ones((100, 100))
        
        # Act
        result = denoise_image(image, method='bilateral')
        
        # Assert
        mock_bilateral.assert_called_once()
    
    @patch('ocr.preprocess.cv2.GaussianBlur')
    def test_denoise_gaussian(self, mock_gaussian):
        """Test Gaussian denoising."""
        # Arrange
        image = np.ones((100, 100))
        mock_gaussian.return_value = np.ones((100, 100))
        
        # Act
        result = denoise_image(image, method='gaussian')
        
        # Assert
        mock_gaussian.assert_called_once()
    
    def test_denoise_unknown_method(self):
        """Test error handling for unknown denoise method."""
        # Arrange
        image = np.ones((100, 100))
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            denoise_image(image, method='unknown')
        assert "Unknown denoising method" in str(exc_info.value)


class TestPreprocessingPipeline:
    
    @patch('ocr.preprocess.load_image')
    @patch('ocr.preprocess.pil_to_cv2')
    @patch('ocr.preprocess.to_grayscale')
    @patch('ocr.preprocess.denoise_image')
    @patch('ocr.preprocess.enhance_contrast')
    @patch('ocr.preprocess.resize_image')
    @patch('ocr.preprocess.apply_threshold')
    @patch('ocr.preprocess.Image.fromarray')
    def test_preprocess_image_full_pipeline(self, mock_fromarray, mock_threshold, 
                                          mock_resize, mock_enhance, mock_denoise,
                                          mock_grayscale, mock_pil_to_cv2, mock_load):
        """Test full preprocessing pipeline."""
        # Arrange
        mock_pil_image = Mock(spec=Image.Image)
        mock_load.return_value = mock_pil_image
        mock_cv_image = np.ones((100, 100, 3))
        mock_pil_to_cv2.return_value = mock_cv_image
        
        mock_grayscale.return_value = np.ones((100, 100))
        mock_denoise.return_value = np.ones((100, 100))
        mock_enhance.return_value = np.ones((100, 100))
        mock_resize.return_value = np.ones((200, 200))
        mock_threshold.return_value = np.ones((200, 200))
        
        mock_result_image = Mock(spec=Image.Image)
        mock_fromarray.return_value = mock_result_image
        
        # Act
        result = preprocess_image("test.jpg")
        
        # Assert
        assert result == mock_result_image
        mock_load.assert_called_once_with("test.jpg")
        mock_grayscale.assert_called_once()
        mock_denoise.assert_called_once()
        mock_enhance.assert_called_once()
        mock_resize.assert_called_once()
        mock_threshold.assert_called_once()
    
    @patch('ocr.preprocess.preprocess_image')
    def test_preprocess_for_text_type_document(self, mock_preprocess):
        """Test document-optimized preprocessing."""
        # Arrange
        mock_result = Mock(spec=Image.Image)
        mock_preprocess.return_value = mock_result
        
        # Act
        result = preprocess_for_text_type("test.jpg", text_type='document')
        
        # Assert
        assert result == mock_result
        mock_preprocess.assert_called_once()
        # Verify the call was made with document-optimized parameters
        call_args = mock_preprocess.call_args
        assert call_args[1]['threshold_method'] == 'otsu'
        assert call_args[1]['resize_scale'] == 2.0
    
    def test_preprocess_for_text_type_unknown(self):
        """Test error handling for unknown text type."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            preprocess_for_text_type("test.jpg", text_type='unknown')
        assert "Unknown text type" in str(exc_info.value)


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy array image for testing."""
    return np.ones((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image for testing."""
    return Image.new('RGB', (100, 100), color='white')


class TestIntegrationWithRealData:
    """Integration tests with actual image data."""
    
    def test_resize_image_real_data(self, sample_numpy_image):
        """Test image resizing with real numpy array."""
        # Act
        result = resize_image(sample_numpy_image, scale_factor=2.0)
        
        # Assert
        assert result.shape == (200, 200, 3)
    
    def test_enhance_contrast_real_data(self, sample_numpy_image):
        """Test contrast enhancement with real numpy array."""
        # Act
        result = enhance_contrast(sample_numpy_image, alpha=1.5, beta=10)
        
        # Assert
        assert result.shape == sample_numpy_image.shape
        assert result.dtype == sample_numpy_image.dtype