import requests
import base64
from PIL import Image
import io

# Create a simple test image
img = Image.new('RGB', (400, 200), color=(255, 255, 255))
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)

# Add some text
text = """AADHAAR
1234 5678 9012
Name: Rahul Kumar
DOB: 15/08/1990
Gender: Male"""

draw.text((50, 50), text, fill=(0, 0, 0))

# Save to bytes
buffer = io.BytesIO()
img.save(buffer, format='PNG')
buffer.seek(0)

# Test upload
print("Testing upload to http://localhost:5555/api/process-for-redaction")
try:
    files = {'file': ('test_aadhaar.png', buffer, 'image/png')}
    response = requests.post('http://localhost:5555/api/process-for-redaction', files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    print(response.text[:500])
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success!")
        print(f"Document Type: {data.get('document_type')}")
        print(f"Fields Detected: {len(data.get('detected_fields', []))}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
