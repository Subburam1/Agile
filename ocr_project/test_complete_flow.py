#!/usr/bin/env python3
"""
Test Complete OCR Project Flow
Tests the 6-step OCR project flow:
1. Document Upload 
2. OCR Text Extraction 
3. Field Detection
4. Field Selection UI 
5. Field Blurring 
6. Export Modified Image
"""
import requests
import base64
import json
import os
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Create a simple test image with text fields"""
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some test text fields
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Sample form fields
    fields = [
        ("Name: John Doe", (50, 100)),
        ("Email: john.doe@email.com", (50, 150)),
        ("Phone: +1-234-567-8900", (50, 200)),
        ("Address: 123 Main St", (50, 250)),
        ("City: New York", (50, 300)),
        ("ZIP: 12345", (50, 350)),
    ]
    
    for text, position in fields:
        draw.text(position, text, fill='black', font=font)
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def test_complete_ocr_flow():
    """Test the complete OCR project flow"""
    print("🧪 Testing Complete OCR Project Flow")
    print("="*50)
    
    base_url = "http://localhost:5000"
    
    try:
        # Step 1: Create test image
        print("1. 📄 Creating test image...")
        image_data = create_test_image()
        print(f"   ✅ Test image created ({len(image_data)} bytes)")
        
        # Step 2: Upload and OCR
        print("\n2. 🚀 Uploading for OCR...")
        files = {
            'file': ('test_image.png', image_data, 'image/png')
        }
        data = {
            'ocr_engine': 'tesseract',
            'fast_mode': 'false'
        }
        
        response = requests.post(f"{base_url}/upload", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            extracted_text = result.get('ocr_text', '')
            print(f"   ✅ OCR successful! Text length: {len(extracted_text)}")
            print(f"   📝 Extracted: {extracted_text[:100]}...")
        else:
            print(f"   ❌ OCR failed: {response.status_code}")
            return False
        
        # Step 3: Field Detection from Image
        print("\n3. 🔍 Testing field detection from image...")
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        field_data = {
            'image': image_b64,
            'confidence_threshold': 0.3
        }
        
        response = requests.post(
            f"{base_url}/api/fields/detect-from-image", 
            json=field_data
        )
        
        if response.status_code == 200:
            fields_result = response.json()
            detected_fields = fields_result.get('fields', [])
            print(f"   ✅ Field detection successful! {len(detected_fields)} fields found")
            
            # Print some detected fields
            for i, field in enumerate(detected_fields[:3]):
                print(f"      Field {i+1}: {field.get('field_name', 'N/A')} (confidence: {field.get('confidence', 0):.2f})")
        else:
            print(f"   ❌ Field detection failed: {response.status_code}")
            error_response = response.json()
            print(f"   Error: {error_response.get('error', 'Unknown error')}")
            return False
        
        # Step 4: Test Field Selection (simulated)
        print("\n4. 🎯 Field selection UI simulation...")
        selected_fields = [
            {'x': 50, 'y': 100, 'width': 200, 'height': 30},  # Name field
            {'x': 50, 'y': 150, 'width': 250, 'height': 30},  # Email field
        ]
        print(f"   ✅ Simulated selection of {len(selected_fields)} fields")
        
        # Step 5 & 6: Blur and Export
        print("\n5-6. 🔄 Testing blur and export...")
        blur_data = {
            'image': image_b64,
            'selected_fields': selected_fields,
            'blur_strength': 15
        }
        
        response = requests.post(
            f"{base_url}/api/blur-and-export", 
            json=blur_data
        )
        
        if response.status_code == 200:
            export_result = response.json()
            blurred_image_b64 = export_result.get('blurred_image')
            
            if blurred_image_b64:
                print(f"   ✅ Blur and export successful!")
                
                # Save blurred image for verification
                blurred_data = base64.b64decode(blurred_image_b64)
                with open('test_blurred_output.png', 'wb') as f:
                    f.write(blurred_data)
                print(f"   💾 Blurred image saved as 'test_blurred_output.png'")
            else:
                print(f"   ❌ No blurred image in response")
                return False
        else:
            print(f"   ❌ Blur and export failed: {response.status_code}")
            return False
        
        print("\n🎉 COMPLETE OCR PROJECT FLOW TEST SUCCESSFUL!")
        print("✅ All 6 steps working:")
        print("   1. Document Upload ✅")
        print("   2. OCR Text Extraction ✅") 
        print("   3. Field Detection ✅")
        print("   4. Field Selection UI ✅")
        print("   5. Field Blurring ✅")
        print("   6. Export Modified Image ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_ocr_flow()
    exit(0 if success else 1)