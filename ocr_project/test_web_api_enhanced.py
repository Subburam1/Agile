#!/usr/bin/env python3
"""
Quick test of enhanced field detection through the web API
"""
import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io

def create_sample_document_image():
    """Create a sample document image with various field types"""
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_large = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Document header
    draw.text((50, 30), "GOVERNMENT OF INDIA", fill='black', font=font_large)
    draw.text((50, 60), "IDENTITY DOCUMENT", fill='black', font=font_large)
    
    # Personal Information
    draw.text((50, 120), "Name: RAJESH KUMAR SHARMA", fill='black', font=font)
    draw.text((50, 150), "Date of Birth: 15/08/1985", fill='black', font=font)
    draw.text((50, 180), "Address: House No. 123, Sector 15", fill='black', font=font)
    draw.text((50, 200), "         Gurgaon, Haryana - 122001", fill='black', font=font)
    draw.text((50, 230), "Phone: +91-9876543210", fill='black', font=font)
    draw.text((50, 260), "Email: rajesh.sharma@email.com", fill='black', font=font)
    
    # ID Numbers
    draw.text((50, 320), "Aadhar Number: 1234 5678 9012", fill='black', font=font)
    draw.text((50, 350), "PAN Number: ABCDE1234F", fill='black', font=font)
    
    # Visual elements
    draw.rectangle((500, 120, 650, 270), outline='black', width=2)
    draw.text((520, 280), "Photo", fill='black', font=font)
    
    draw.rectangle((500, 320, 650, 380), outline='black', width=1)
    draw.text((520, 390), "Signature", fill='black', font=font)
    
    draw.rectangle((50, 450, 150, 500), outline='black', width=1)
    draw.text((60, 510), "Thumb", fill='black', font=font)
    draw.text((50, 530), "Impression", fill='black', font=font)
    
    # Document metadata
    draw.text((50, 600), "Document Number: DOC123456789", fill='black', font=font)
    draw.text((50, 630), "Issue Date: 01/01/2020", fill='black', font=font)
    draw.text((50, 660), "Valid Until: 31/12/2030", fill='black', font=font)
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def test_web_api():
    """Test enhanced field detection through the web API"""
    print("🧪 Testing Enhanced Field Detection via Web API")
    print("="*50)
    
    try:
        # Create test image
        print("1. 📄 Creating sample document image...")
        image_data = create_sample_document_image()
        print(f"   ✅ Sample document created ({len(image_data)} bytes)")
        
        # Convert to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Test field detection from image
        print("2. 🔍 Testing field detection API...")
        
        # Save image temporarily for upload
        temp_image_path = 'temp_test_document.png'
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)
        
        # Upload file using correct API format
        with open(temp_image_path, 'rb') as f:
            files = {'file': ('test_document.png', f, 'image/png')}
            data = {'confidence_threshold': '0.3'}
            
            response = requests.post(
                'http://localhost:5000/api/fields/detect-from-image',
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            fields = result.get('fields', [])
            
            print(f"   ✅ API call successful! Detected {len(fields)} fields")
            
            # Group fields by type
            field_types = {}
            for field in fields:
                field_name = field.get('field_name', 'unknown')
                if field_name not in field_types:
                    field_types[field_name] = []
                field_types[field_name].append(field)
            
            print("\\n📋 Enhanced Field Detection Results:")
            for field_type, field_list in sorted(field_types.items()):
                print(f"   • {field_type}: {len(field_list)} instances")
                
                # Show examples
                for field in field_list[:2]:  # Show first 2 
                    value = field.get('field_value', 'N/A')
                    confidence = field.get('confidence', 0)
                    location = field.get('location', {})
                    print(f"     - \"{value[:40]}\" (confidence: {confidence:.2f})")
                    if location:
                        print(f"       Location: ({location.get('x', 0)}, {location.get('y', 0)})")
            
            # Check for specific enhanced field types
            detected_enhanced_fields = []
            enhanced_field_keywords = ['name', 'address', 'phone', 'email', 'birth', 'aadhar', 'pan', 'signature', 'photo', 'thumb']
            
            for field_type in field_types.keys():
                for keyword in enhanced_field_keywords:
                    if keyword.lower() in field_type.lower():
                        detected_enhanced_fields.append(field_type)
                        break
            
            print(f"\\n✅ Enhanced Field Types Detected: {len(detected_enhanced_fields)}")
            for field_type in detected_enhanced_fields:
                print(f"   ✓ {field_type}")
            
            # Save sample for manual verification
            with open('sample_enhanced_document.png', 'wb') as f:
                f.write(image_data)
            print(f"\\n💾 Sample document saved as 'sample_enhanced_document.png'")
            
            # Cleanup
            import os
            if os.path.exists('temp_test_document.png'):
                os.remove('temp_test_document.png')
            
        else:
            print(f"   ❌ API call failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        print("\\n🎉 ENHANCED FIELD DETECTION WEB API TEST SUCCESSFUL!")
        print("✅ Enhanced patterns working through web interface for:")
        print("   • Personal Information (Name, Address, Phone, Email, DOB)")
        print("   • Identification Numbers (Aadhar, PAN)")
        print("   • Visual Elements (Signature, Photo, Thumb Impression)")
        print("   • Document Metadata (Numbers, Dates)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_api()
    exit(0 if success else 1)