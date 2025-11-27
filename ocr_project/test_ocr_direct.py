# -*- coding: utf-8 -*-
import pytesseract
from PIL import Image
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Test OCR on the image directly
img_path = 'test_document_quality.png'
img = Image.open(img_path)

print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
print("\n" + "=" * 80)
print("OCR TEXT EXTRACTION:")
print("=" * 80)

text = pytesseract.image_to_string(img)
print(f"Extracted text ({len(text)} characters):\n")
print(text)
print("\n" + "=" * 80)
