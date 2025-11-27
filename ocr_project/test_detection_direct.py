# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'd:/Agile/ocr_project')

from simple_redaction_server import detect_sensitive_fields

# Test the detection function directly
test_text = """PERSONAL DETAILS FORM

APPLICANT INFORMATION

Name: Rajesh Kumar Sharma
Father's Name: Vijay Kumar Sharma
Date of Birth: 15/08/1990

Gender: Male

RESIDENTIAL ADDRESS

Address: House No 234, Green Park Extension, Sector 12, New Delhi
Pincode: 110016

State: Delhi

CONTACT INFORMATION
Mobile: 9876543210

Email: rajesh.sharma@email.com
Phone: 8765432109

IDENTIFICATION NUMBERS
Aadhaar: 1234 5678 9012
PAN: ABCDE1234F

Bank Account: 12345678901
Passport: M1234567

OTHER DETAILS (Testing Context Detection)

Reference Number: 9988776655
Transaction ID: 110025
Order Number: 556677

PASSWORD INFORMATION

Password: MySecret123!
PIN: 1234
"""

print("=" * 80)
print("TESTING FIELD DETECTION PATTERNS")
print("=" * 80)

# Test with image path
image_path = 'd:/Agile/ocr_project/test_document_quality.png'
detected_fields = detect_sensitive_fields(test_text, image_path)

print(f"\nTotal fields detected: {len(detected_fields)}\n")

# Organize by category
categories = {}
for field in detected_fields:
    cat = field['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(field)

for category, fields in sorted(categories.items()):
    print(f"\n{category.upper().replace('_', ' ')}:")
    print("-" * 80)
    for field in fields:
        print(f"  {field['field_name']}: {field['field_value']}")
        print(f"    Confidence: {field['confidence']}, Auto-selected: {field['auto_selected']}")

print("\n" + "=" * 80)
print("CONTEXT-AWARE PATTERN TEST RESULTS:")
print("=" * 80)

# Check specific fields
mobile_found = any(f['field_name'] == 'Mobile' and '9876543210' in f['field_value'] for f in detected_fields)
pincode_found = any(f['field_name'] == 'Pincode' and '110016' in f['field_value'] for f in detected_fields)
phone_found = any(f['field_name'] == 'Mobile' and '8765432109' in f['field_value'] for f in detected_fields)

# Check for false positives
ref_as_mobile = any('9988776655' in f['field_value'] and f['field_name'] == 'Mobile' for f in detected_fields)
trans_as_pincode = any('110025' in f['field_value'] and f['field_name'] == 'Pincode' for f in detected_fields)

print(f"\n[EXPECTED] Mobile: 9876543210 detected: {mobile_found}")
print(f"[EXPECTED] Phone: 8765432109 detected: {phone_found}")  
print(f"[EXPECTED] Pincode: 110016 detected: {pincode_found}")
print(f"\n[FALSE POSITIVE CHECK] Reference 9988776655 detected as Mobile: {ref_as_mobile}")
print(f"[FALSE POSITIVE CHECK] Transaction 110025 detected as Pincode: {trans_as_pincode}")

print("\n" + "=" * 80)
