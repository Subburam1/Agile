"""Tests for OCR functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile

from ocr.ocr import extract_text, extract_text_from_file, get_available_languages


class TestOCRExtraction:
    
    @patch('ocr.ocr.pytesseract.image_to_string')
    def test_extract_text_with_pil_image(self, mock_tesseract):
        """Test text extraction with PIL Image input."""
        # Arrange
        mock_tesseract.return_value = "  Sample text  "
        mock_image = Mock(spec=Image.Image)
        
        # Act
        result = extract_text(mock_image)
        
        # Assert
        assert result == "Sample text"
        mock_tesseract.assert_called_once_with(mock_image, lang='eng', config='--psm 6')
    
    @patch('ocr.ocr.Image.open')
    @patch('ocr.ocr.pytesseract.image_to_string')
    def test_extract_text_with_file_path(self, mock_tesseract, mock_image_open):
        """Test text extraction with file path input."""
        # Arrange
        mock_tesseract.return_value = "Text from file"
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image
        
        # Act
        result = extract_text("test_image.jpg")
        
        # Assert
        assert result == "Text from file"
        mock_image_open.assert_called_once_with("test_image.jpg")
        mock_tesseract.assert_called_once_with(mock_image, lang='eng', config='--psm 6')
    
    @patch('ocr.ocr.pytesseract.image_to_string')
    def test_extract_text_with_custom_lang_and_config(self, mock_tesseract):
        """Test text extraction with custom language and config."""
        # Arrange
        mock_tesseract.return_value = "German text"
        mock_image = Mock(spec=Image.Image)
        
        # Act
        result = extract_text(mock_image, lang='deu', config='--psm 8')
        
        # Assert
        assert result == "German text"
        mock_tesseract.assert_called_once_with(mock_image, lang='deu', config='--psm 8')
    
    @patch('ocr.ocr.Image.open')
    def test_extract_text_file_not_found(self, mock_image_open):
        """Test FileNotFoundError handling."""
        # Arrange
        mock_image_open.side_effect = FileNotFoundError("No such file")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_text("nonexistent.jpg")
        assert "Image file not found" in str(exc_info.value)
    
    @patch('ocr.ocr.pytesseract.image_to_string')
    def test_extract_text_tesseract_not_found(self, mock_tesseract):
        """Test TesseractNotFoundError handling."""
        # Arrange
        from pytesseract import TesseractNotFoundError
        mock_tesseract.side_effect = TesseractNotFoundError("Tesseract not found")
        mock_image = Mock(spec=Image.Image)
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            extract_text(mock_image)
        assert "Tesseract not found" in str(exc_info.value)
    
    @patch('ocr.ocr.extract_text')
    @patch('ocr.ocr.preprocess_image')
    def test_extract_text_from_file_with_preprocessing(self, mock_preprocess, mock_extract):
        """Test extract_text_from_file with preprocessing enabled."""
        # Arrange
        mock_processed_image = Mock(spec=Image.Image)
        mock_preprocess.return_value = mock_processed_image
        mock_extract.return_value = "Processed text"
        
        # Act
        result = extract_text_from_file("test.jpg", preprocess=True)
        
        # Assert
        assert result == "Processed text"
        mock_preprocess.assert_called_once_with("test.jpg")
        mock_extract.assert_called_once_with(mock_processed_image, lang='eng', config='--psm 6')
    
    @patch('ocr.ocr.extract_text')
    def test_extract_text_from_file_without_preprocessing(self, mock_extract):
        """Test extract_text_from_file without preprocessing."""
        # Arrange
        mock_extract.return_value = "Raw text"
        
        # Act
        result = extract_text_from_file("test.jpg", preprocess=False)
        
        # Assert
        assert result == "Raw text"
        mock_extract.assert_called_once_with("test.jpg", lang='eng', config='--psm 6')
    
    @patch('ocr.ocr.pytesseract.get_languages')
    def test_get_available_languages_success(self, mock_get_langs):
        """Test successful language retrieval."""
        # Arrange
        mock_get_langs.return_value = ['eng', 'deu', 'fra']
        
        # Act
        result = get_available_languages()
        
        # Assert
        assert result == ['eng', 'deu', 'fra']
    
    @patch('ocr.ocr.pytesseract.get_languages')
    def test_get_available_languages_fallback(self, mock_get_langs):
        """Test language retrieval fallback on error."""
        # Arrange
        mock_get_langs.side_effect = Exception("Error")
        
        # Act
        result = get_available_languages()
        
        # Assert
        assert result == ['eng']


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple test image
    img = Image.new('RGB', (100, 30), color='white')
    return img


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_image.save(f.name)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestOCRIntegration:
    """Integration tests that require actual images."""
    
    def test_extract_text_real_image(self, temp_image_file):
        """Test with a real temporary image file."""
        try:
            # This might fail if Tesseract is not installed, but that's expected
            result = extract_text(temp_image_file)
            assert isinstance(result, str)
        except Exception as e:
            # Expected if Tesseract is not properly configured
            assert "Tesseract" in str(e) or "not found" in str(e).lower()