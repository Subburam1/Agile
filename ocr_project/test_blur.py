  # -*- coding: utf-8 -*-
import requests
import base64
import json

# Read and encode the test image
with open('test_document_quality.png', 'rb') as f:
    b64_image = base64.b64encode(f.read()).decode()

print("Step 1: Upload and detect fields...")
# Send to API for field detection
response = requests.post(
    'http://localhost:5555/api/process-for-redaction',
    files={'file': open('test_document_quality.png', 'rb')}
)

result = response.json()
print(f"✓ Detected {len(result['detected_fields'])} fields")

# Select a few fields to blur
selected_fields = result['detected_fields'][:3]  # First 3 fields
print(f"\nStep 2: Blurring {len(selected_fields)} fields:")
for f in selected_fields:
    print(f"  - {f['field_name']}: {f['field_value']}")
    print(f"    Coordinates: x={f['coordinates']['x']:.1f}%, y={f['coordinates']['y']:.1f}%, w={f['coordinates']['width']:.1f}%, h={f['coordinates']['height']:.1f}%")

# Call blur API
print("\nStep 3: Calling blur API...")
blur_payload = {
    'image_data': result['image_data'],
    'selected_fields': [f['coordinates'] for f in selected_fields],
    'blur_strength': 25
}

blur_response = requests.post(
    'http://localhost:5555/api/blur-and-export',
    json=blur_payload
)

blur_result = blur_response.json()

if blur_result.get('success'):
    print("✓ Blur API succeeded")
    
    # Save the blurred image
    blurred_data = blur_result['blurred_image'].split(',')[1]
    with open('test_blurred_output.png', 'wb') as f:
        f.write(base64.b64decode(blurred_data))
    
    print("✓ Saved blurred image to test_blurred_output.png")
    print("\nPlease check the image to verify blur quality!")
else:
    print(f"✗ Blur API failed: {blur_result.get('error')}")
