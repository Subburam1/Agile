# -*- coding: utf-8 -*-
import requests
import base64
import json
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Read and encode the test image
with open('test_document_quality.png', 'rb') as f:
    b64_image = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post(
    'http://localhost:5555/api/process-for-redaction',
    json={'image': b64_image}
)

# Print formatted results
result = response.json()
print("=" * 80)
print("FIELD DETECTION RESULTS")
print("=" * 80)
print(f"\nTotal fields detected: {len(result.get('detected_fields', []))}")
print(f"OCR Text extracted: {len(result.get('ocr_text', ''))} characters\n")

print("\nDETECTED FIELDS:")
print("-" * 80)

for field in result.get('detected_fields', []):
    print(f"\n{field['field_name']}:")
    print(f"  Value: {field['field_value']}")
    print(f"  Confidence: {field['confidence']}")
    print(f"  Category: {field['category']}")
    print(f"  Auto-selected: {field['auto_selected']}")
    print(f"  Coordinates: x={field['coordinates']['x']:.1f}%, y={field['coordinates']['y']:.1f}%, w={field['coordinates']['width']:.1f}%, h={field['coordinates']['height']:.1f}%")

print("\n" + "=" * 80)
print("CONTEXT-AWARE DETECTION TEST:")
print("=" * 80)

# Check specific fields
mobile_found = any(f['field_name'] == 'Mobile' for f in result.get('detected_fields', []))
pincode_found = any(f['field_name'] == 'Pincode' for f in result.get('detected_fields', []))
ref_num_detected = any('9988776655' in f['field_value'] for f  in result.get('detected_fields', []))
trans_id_detected = any('110025' in f['field_value'] for f in result.get('detected_fields', []))

print(f"\n[PASS] Mobile (9876543210) with 'Mobile:' label - Detected: {mobile_found}")
print(f"[PASS] Pincode (110016) with 'Pincode:' label - Detected: {pincode_found}")
print(f"[FAIL] Reference Number (9988776655) without label - Detected as sensitive: {ref_num_detected}")
print(f"[FAIL] Transaction ID (110025) without label - Detected as pincode: {trans_id_detected}")


print("\n" + "=" * 80)
