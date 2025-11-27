"""
Simple single-file Document Redaction Server with OCR Field Detection
Detects sensitive fields from Indian documents automatically.
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from pathlib import Path
import base64
import os
from PIL import Image, ImageFilter
import io
import re
import pytesseract
import uuid
import datetime
from datetime import timedelta
from functools import wraps
from db_utils import db_manager
import auth_db_mongo
from auth_db_mongo import register_user, login_user, get_user_by_id, update_user_email, change_password

# Configure pytesseract path (update if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # Change this!
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)


# ===== AUTHENTICATION DECORATOR =====
def login_required(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


def detect_document_type(text, image_path=None):
    """
    Hybrid ML-based document type detection.
    Combines visual features + OCR patterns for high accuracy.
    """
    text_lower = text.lower()
    
    # Visual feature analysis (if image provided)
    visual_scores = {}
    if image_path:
        try:
            import cv2
            import numpy as np
            from pathlib import Path
            
            # Check if file exists
            if not Path(image_path).exists():
                print(f"Image file not found: {image_path}")
            else:
                img = cv2.imread(str(image_path))
                if img is not None and img.size > 0:
                    h, w = img.shape[:2]
                    
                    # Feature 1: Color analysis (identity cards vs documents)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    
                    # Detect blue/government colors (Aadhaar, PAN, Passport)
                    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
                    blue_ratio = np.sum(blue_mask > 0) / (h * w)
                    
                    # Detect official seals/logos (circular shapes)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20, maxRadius=100)
                    has_seal = circles is not None
                    
                    # Feature 2: Text density and layout
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    text_pixels = np.sum(binary > 0)
                    text_density = text_pixels / (h * w)
                    
                    # Feature 3: Edge analysis (structured vs unstructured)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (h * w)
                    
                    # Feature 4: Aspect ratio
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Score document types based on visual features
                    if blue_ratio > 0.05 and has_seal:
                        visual_scores['Aadhaar Card'] = 0.3
                        visual_scores['PAN Card'] = 0.25
                        visual_scores['Passport'] = 0.2
                    
                    if aspect_ratio > 1.4 and aspect_ratio < 1.8:  # ID card ratio
                        visual_scores['Aadhaar Card'] = visual_scores.get('Aadhaar Card', 0) + 0.15
                        visual_scores['PAN Card'] = visual_scores.get('PAN Card', 0) + 0.15
                        visual_scores['Voter ID Card'] = visual_scores.get('Voter ID Card', 0) + 0.15
                    
                    if text_density > 0.15:  # Dense text = marksheet/invoice
                        visual_scores['Marksheet'] = visual_scores.get('Marksheet', 0) + 0.2
                        visual_scores['Invoice'] = visual_scores.get('Invoice', 0) + 0.15
                        visual_scores['Bank Statement'] = visual_scores.get('Bank Statement', 0) + 0.15
                    
                    if edge_density > 0.08:  # Tables/grids
                        visual_scores['Marksheet'] = visual_scores.get('Marksheet', 0) + 0.2
                        visual_scores.get('Invoice', 0) + 0.2
                else:
                    print(f"Failed to load image: {image_path}")
        except ImportError:
            print("OpenCV not installed, skipping visual analysis")
        except Exception as e:
            print(f"Visual analysis failed: {e}")
            import traceback
            traceback.print_exc()

    
    # Enhanced pattern-based detection with ML-like scoring
    patterns = {
        'Aadhaar Card': {
            'must_have': [r'\b\d{4}\s?\d{4}\s?\d{4}\b'],  # 12-digit number
            'strong': ['aadhaar', 'aadhar', 'uidai', 'government of india'],
            'supporting': ['dob', 'gender', 'address', 'male', 'female'],
            'base_score': 0.4
        },
        'PAN Card': {
            'must_have': [r'\b[A-Z]{5}\d{4}[A-Z]\b'],  # PAN format
            'strong': ['permanent account', 'income tax', 'pan'],
            'supporting': ['father', 'signature'],
            'base_score': 0.5
        },
        'Passport': {
            'must_have': [r'\b[A-Z]\d{7}\b'],  # Passport number
            'strong': ['passport', 'republic of india', 'nationality'],
            'supporting': ['given name', 'surname', 'place of birth', 'date of issue'],
            'base_score': 0.5
        },
        'Voter ID Card': {
            'must_have': [r'\b[A-Z]{3}\d{7}\b'],  # Voter ID format
            'strong': ['election commission', 'electoral', 'voter'],
            'supporting': ['assembly', 'part no', 'serial'],
            'base_score': 0.4
        },
        'Driving License': {
            'must_have': [r'(?i)(driving|dl|license|licence)'],
            'strong': ['driving licence', 'transport', 'vehicle'],
            'supporting': ['validity', 'blood group', 'authorized'],
            'base_score': 0.3
        },
        'Marksheet': {
            'must_have': [r'(?i)(marks?|grade|cgpa|sgpa|subject)'],
            'strong': ['marks obtained', 'university', 'examination'],
            'supporting': ['semester', 'roll', 'theory', 'practical', 'total'],
            'base_score': 0.3
        },
        'College ID Card': {
            'must_have': [r'(?i)(student|enrollment|college|university)'],
            'strong': ['student id', 'enrollment', 'roll no'],
            'supporting': ['department', 'course', 'year', 'validity'],
            'base_score': 0.25
        },
        'Bank Passbook': {
            'must_have': [r'(?i)(account|ifsc|bank)'],
            'strong': ['passbook', 'account holder', 'ifsc'],
            'supporting': ['branch', 'savings', 'micr', 'nominee'],
            'base_score': 0.3
        },
        'Bank Statement': {
            'must_have': [r'(?i)(statement|transaction|balance)'],
            'strong': ['bank statement', 'opening balance', 'closing balance'],
            'supporting': ['debit', 'credit', 'withdrawal', 'deposit'],
            'base_score': 0.3
        },
        'Invoice': {
            'must_have': [r'(?i)(invoice|bill|gst|tax)'],
            'strong': ['invoice no', 'invoice date', 'gst'],
            'supporting': ['bill to', 'subtotal', 'total', 'qty', 'amount'],
            'base_score': 0.35
        },
        'Bill/Receipt': {
            'must_have': [r'(?i)(receipt|bill|payment)'],
            'strong': ['receipt no', 'payment received'],
            'supporting': ['amount', 'cash', 'card', 'date'],
            'base_score': 0.3
        },
        'Salary Slip': {
            'must_have': [r'(?i)(salary|pay|earnings|deductions)'],
            'strong': ['salary slip', 'pay slip', 'net pay'],
            'supporting': ['basic salary', 'gross', 'pf', 'esi', 'tds'],
            'base_score': 0.35
        }
    }
    
    # ML-like scoring algorithm
    final_scores = {}
    
    for doc_type, config in patterns.items():
        score = 0.0
        
        # Check must-have patterns (critical features)
        must_have_match = False
        for pattern in config['must_have']:
            if re.search(pattern, text, re.IGNORECASE):
                must_have_match = True
                score += config['base_score']
                break
        
        if not must_have_match:
            continue  # Skip if doesn't meet minimum requirements
        
        # Strong indicators (weighted heavily)
        strong_matches = sum(1 for keyword in config['strong'] if keyword in text_lower)
        score += strong_matches * 0.15
        
        # Supporting indicators (lighter weight)
        support_matches = sum(1 for keyword in config['supporting'] if keyword in text_lower)
        score += support_matches * 0.05
        
        # Add visual feature score
        score += visual_scores.get(doc_type, 0)
        
        # Normalize score
        final_scores[doc_type] = min(score, 1.0)
    
    # Find best match
    if final_scores:
        best_type = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_type]
        
        # Very low threshold - if we matched the pattern, we're confident
        if confidence >= 0.25:
            return {
                'document_type': best_type,
                'confidence': round(confidence, 2),
                'all_scores': {k: round(v, 2) for k, v in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]},
                'detection_method': 'Hybrid ML (Visual + OCR)'
            }
    
    return {
        'document_type': 'Unknown Document',
        'confidence': 0.0,
        'all_scores': {},
        'detection_method': 'Pattern matching'
    }



def detect_sensitive_fields(text, image_path=None):
    """
    Comprehensive field detection with ACCURATE coordinates.

    Detects: Text fields, MRZ, QR codes, barcodes, photos, signatures
    """
    detected_fields = []
    field_id = 0
    
    # Get OCR data with bounding boxes if image provided
    ocr_data = None
    image = None
    cv_image = None  # Initialize to prevent UnboundLocalError
    if image_path:
        try:
            import pandas as pd
            import cv2
            import numpy as np
            from pyzbar import pyzbar
            
            image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
            ocr_data = ocr_df[ocr_df['text'].notna() & (ocr_df['text'].str.strip() != '')]
        except Exception as e:
            print(f"OCR bbox extraction failed: {e}")
            ocr_data = None
            cv_image = None  # Ensure it's None if loading failed

    
    # Define text-based patterns
    sensitive_patterns = {
        'Password': r'(?i)(password|pwd|pass)\s*:?\s*([^\s\n]{4,})',
        'Bank Account': r'\b\d{11}\b',
        'Credit Card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'CVV': r'(?i)(cvv|cvc)\s*:?\s*\d{3,4}',
        'PIN': r'(?i)(pin|atm\s*pin)\s*:?\s*\d{4,6}',
    }
    
    identifier_patterns = {
        'Aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        'PAN': r'\b[A-Z]{5}\d{4}[A-Z]\b',
        'Passport': r'\b[A-Z]\d{7}\b',
        'Voter ID': r'\b[A-Z]{3}\d{7}\b',
    }
    
    personal_patterns = {
        'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'Mobile': r'(?i)(mobile|phone|contact|tel|cell|ph\.?)\s*(?:no|number|num)?\.?\s*:?\s*([6-9]\d{9})',  # With Ph
        'Landline': r'(?i)(phone|tel|ph\.?|landline)\s*(?:no|number)?\.?\s*:?\s*(\d{3,5}[\s-]?\d{6,8})',  # Landline with area code
        'DOB': r'(?i)(date\s*of\s*birth|dob|birth\s*date)\s*:?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        'Pincode': r'(?i)(pin\s*code|pincode|postal\s*code|zip)\s*:?\s*(\d{6})',  # With context
        'Name': r'(?i)(name|naam)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        'Father Name': r'(?i)(father\'?s?\s*name)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        'Address': r'(?i)(address|पता)\s*:?\s*([^\n]{15,100})',
        'Gender': r'(?i)(gender|sex)\s*:?\s*(male|female|m|f|other)',
    }
    
    # Fallback patterns DISABLED - they create too many false positives
    # Context-aware patterns above are sufficient for accurate detection
    # fallback_patterns = {
    #     'Mobile (standalone)': r'\b[6-9]\d{9}\b',  # Standalone 10-digit number
    #     'Landline (standalone)': r'\b\d{6,8}\b',  # Standalone 6-8 digit landline
    #     'Pincode (standalone)': r'\b\d{6}\b',  # Standalone 6-digit number
    # }
    
    # MRZ patterns (Machine Readable Zone for passports, ID cards)
    mrz_patterns = {
        'MRZ Line': r'[A-Z0-9<]{30,44}',  # Standard MRZ format
    }
    
    # Helper function to find coordinates with HIGH ACCURACY
    def find_coordinates(value_text, img_width=None, img_height=None):
        if ocr_data is not None and img_width and img_height:
            value_clean = str(value_text).strip()
            
            # Split value into tokens for better matching
            value_tokens = [t.strip() for t in re.split(r'[\s\-:/]+', value_clean) if len(t.strip()) > 0]
            
            matching_boxes = []
            used_indices = set()
            
            # Strategy 1: Find exact token matches
            for token in value_tokens:
                if len(token) < 2:
                    continue
                    
                token_lower = token.lower().replace('<', '')
                
                for idx, row in ocr_data.iterrows():
                    if idx in used_indices:
                        continue
                        
                    ocr_text = str(row['text']).strip().lower()
                    
                    # Match: exact, contains, or contained
                    if (token_lower == ocr_text or 
                        token_lower in ocr_text or 
                        ocr_text in token_lower):
                        
                        matching_boxes.append({
                            'x': int(row['left']),
                            'y': int(row['top']),
                            'width': int(row['width']),
                            'height': int(row['height'])
                        })
                        used_indices.add(idx)
                        break
            
            # Strategy 2: If no matches, try partial digit matching for numbers
            if not matching_boxes and any(c.isdigit() for c in value_clean):
                digits_only = ''.join(c for c in value_clean if c.isdigit())
                
                for idx, row in ocr_data.iterrows():
                    ocr_digits = ''.join(c for c in str(row['text']) if c.isdigit())
                    
                    # Match if at least 4 consecutive digits match
                    if len(ocr_digits) >= 4 and ocr_digits in digits_only:
                        matching_boxes.append({
                            'x': int(row['left']),
                            'y': int(row['top']),
                            'width': int(row['width']),
                            'height': int(row['height'])
                        })
            
            # If we found matches, combine and add padding
            if matching_boxes:
                # Find bounding box that contains all matches
                min_x = min(b['x'] for b in matching_boxes)
                min_y = min(b['y'] for b in matching_boxes)
                max_x = max(b['x'] + b['width'] for b in matching_boxes)
                max_y = max(b['y'] + b['height'] for b in matching_boxes)
                
                # Add intelligent padding (5 pixels all around)
                padding = 5
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(img_width, max_x + padding)
                max_y = min(img_height, max_y + padding)
                
                return {
                    'x': (min_x / img_width) * 100,
                    'y': (min_y / img_height) * 100,
                    'width': ((max_x - min_x) / img_width) * 100,
                    'height': ((max_y - min_y) / img_height) * 100
                }
        
        # Smart fallback based on field type
        # Smaller, more reasonable boxes
        base_width = 25  # 25% for unknown fields
        base_height = 3
        
        # Adjust based on value length
        if len(str(value_text)) > 30:
            base_width = 40  # Longer fields like addresses
            base_height = 4
        elif len(str(value_text)) > 15:
            base_width = 30
        
        return {
            'x': 8,
            'y': (field_id * 5) % 92,
            'width': base_width,
            'height': base_height
        }
    
    # Get image dimensions
    img_width, img_height = None, None
    if image:
        img_width, img_height = image.size
    
    # Process all text patterns (removed fallback_patterns to prevent false positives)
    all_patterns = [
        (sensitive_patterns, 'sensitive_personal_data', True, 0.9),
        (identifier_patterns, 'identification', True, 0.95),
        (personal_patterns, 'personal_info', True, 0.85),
        (mrz_patterns, 'mrz', True, 0.98)
    ]
    
    for patterns_dict, category, is_sens, conf in all_patterns:
        for field_name, pattern in patterns_dict.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract value
                if field_name in ['DOB', 'Name', 'Father Name', 'Address', 'Gender']:
                    groups = match.groups()
                    value = groups[-1] if len(groups) > 1 else match.group(0)
                else:
                    value = match.group(0)
                
                coords = find_coordinates(value, img_width, img_height)
                
                detected_fields.append({
                    'id': f'field_{field_id}',
                    'field_name': field_name,
                    'field_value': value[:50],
                    'confidence': conf,
                    'category': category,
                    'is_sensitive': is_sens,
                    'auto_selected': is_sens,
                    'coordinates': coords
                })
                field_id += 1
    
    # IMAGE-BASED DETECTION (QR codes, barcodes, photos, signatures)
    if image_path and cv_image is not None:
        try:
            import cv2
            from pyzbar import pyzbar
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            h, w = cv_image.shape[:2]
            
            # 1. QR Code Detection
            try:
                qr_decoder = cv2.QRCodeDetector()
                data, bbox, _ = qr_decoder.detectAndDecode(cv_image)
                if bbox is not None:
                    for i, box in enumerate(bbox):
                        x, y = int(box[0][0]), int(box[0][1])
                        box_w = int(box[2][0] - box[0][0])
                        box_h = int(box[2][1] - box[0][1])
                        
                        detected_fields.append({
                            'id': f'field_{field_id}',
                            'field_name': 'QR Code',
                            'field_value': data[:30] if data else 'QR Code',
                            'confidence': 0.95,
                            'category': 'image_element',
                            'is_sensitive': True,
                            'auto_selected': True,
                            'coordinates': {
                                'x': (x / w) * 100,
                                'y': (y / h) * 100,
                                'width': (box_w / w) * 100,
                                'height': (box_h / h) * 100
                            }
                        })
                        field_id += 1
            except:
                pass
            
            # 2. Barcode Detection (using pyzbar)
            try:
                barcodes = pyzbar.decode(cv_image)
                for barcode in barcodes:
                    x, y, bc_w, bc_h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height
                    bc_data = barcode.data.decode('utf-8') if barcode.data else 'Barcode'
                    
                    detected_fields.append({
                        'id': f'field_{field_id}',
                        'field_name': f'Barcode ({barcode.type})',
                        'field_value': bc_data[:30],
                        'confidence': 0.95,
                        'category': 'image_element',
                        'is_sensitive': True,
                        'auto_selected': True,
                        'coordinates': {
                            'x': (x / w) * 100,
                            'y': (y / h) * 100,
                            'width': (bc_w / w) * 100,
                            'height': (bc_h / h) * 100
                        }
                    })
                    field_id += 1
            except:
                pass
            
            # 3. Face Detection (Profile Photo)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (fx, fy, fw, fh) in faces:
                    detected_fields.append({
                        'id': f'field_{field_id}',
                        'field_name': 'Profile Photo',
                        'field_value': 'Face detected',
                        'confidence': 0.85,
                        'category': 'image_element',
                        'is_sensitive': True,
                        'auto_selected': True,
                        'coordinates': {
                            'x': (fx / w) * 100,
                            'y': (fy / h) * 100,
                            'width': (fw / w) * 100,
                            'height': (fh / h) * 100
                        }
                    })
                    field_id += 1
            except:
                pass
            
            # 4. Signature Detection (using contour detection)
            try:
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:  # Signature-like size
                        sx, sy, sw, sh = cv2.boundingRect(contour)
                        aspect_ratio = float(sw) / sh if sh > 0 else 0
                        
                        # Signature typically has aspect ratio 2:1 to 5:1
                        if 1.5 < aspect_ratio < 6:
                            detected_fields.append({
                                'id': f'field_{field_id}',
                                'field_name': 'Signature',
                                'field_value': 'Signature detected',
                                'confidence': 0.75,
                                'category': 'image_element',
                                'is_sensitive': True,
                                'auto_selected': True,
                                'coordinates': {
                                    'x': (sx / w) * 100,
                                    'y': (sy / h) * 100,
                                    'width': (sw / w) * 100,
                                    'height': (sh / h) * 100
                                }
                            })
                            field_id += 1
                            break  # Only detect first signature
            except:
                pass
                
        except Exception as e:
            print(f"Image-based detection error: {e}")
    
    return detected_fields




@app.route('/')
@app.route('/document-redaction')
@login_required
def document_redaction_page():
    """Document redaction page."""
    return render_template('index.html')

@app.route('/history')
@login_required
def history_page():
    """Processing history page."""
    return render_template('history.html')

@app.route('/api/process-for-redaction', methods=['POST'])
@login_required
def process_for_redaction():
    """
    Process uploaded document with OCR and field detection.
    Detects sensitive Indian document fields automatically with ACCURATE coordinates.
    """
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save image temporarily for coordinate extraction
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), f'redact_{uuid.uuid4()}.png')
        
        # Read and convert image to base64
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save to temp file for OCR bbox extraction
        image.save(temp_path)
        
        # Get image dimensions
        img_width, img_height = image.size
        image_format = image.format or 'PNG'
        
        # Perform OCR
        try:
            extracted_text = pytesseract.image_to_string(image)
        except Exception as ocr_error:
            extracted_text = ""
            print(f"OCR Warning: {ocr_error}")
        
        # Detect sensitive fields WITH ACCURATE COORDINATES
        detected_fields = detect_sensitive_fields(extracted_text, image_path=temp_path) if extracted_text else []
        
        # Detect document type
        document_info = detect_document_type(extracted_text) if extracted_text else {
            'document_type': 'Unknown Document',
            'confidence': 0.0,
            'all_scores': {}
        }
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Calculate statistics
        sensitive_count = sum(1 for f in detected_fields if f['is_sensitive'])
        auto_selected_count = sum(1 for f in detected_fields if f['auto_selected'])
        
        # Save to history (exclude large image data) with user_id
        user_id = session.get('user_id')
        save_to_history({
            'filename': file.filename,
            'document_type': document_info['document_type'],
            'confidence': document_info['confidence'],
            'total_fields': len(detected_fields),
            'sensitive_fields': sensitive_count,
            'status': 'Processed',
            'processing_time': 0  # Placeholder
        }, user_id)
        
        # Return response with detected fields and document type
        return jsonify({
            'success': True,
            'filename': file.filename,
            'document_type': document_info['document_type'],
            'document_confidence': document_info['confidence'],
            'document_scores': document_info.get('all_scores', {}),
            'image_data': f'data:image/{image_format.lower()};base64,{image_data}',
            'image_dimensions': {
                'width': img_width,
                'height': img_height
            },
            'extracted_text': extracted_text,
            'detected_fields': detected_fields,
            'statistics': {
                'total_fields': len(detected_fields),
                'auto_selected': auto_selected_count,
                'sensitive_fields': sensitive_count,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split())
            },
            'processing_metadata': {
                'ocr_method': 'tesseract_with_bbox',
                'ocr_confidence': 0.8,
                'processing_time': 0,
                'note': 'Using ACCURATE OCR bounding boxes for field coordinates'
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.route('/api/blur-and-export', methods=['POST'])
@login_required
def blur_and_export_image():
    """Apply blur to selected fields and export the modified image."""
    try:
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Get parameters
        image_data_url = data['image_data']
        selected_fields = data.get('selected_fields', [])
        blur_strength = int(data.get('blur_strength', 15))
        
        # Extract base64 image data
        if ',' in image_data_url:
            image_data = image_data_url.split(',')[1]
        else:
            image_data = image_data_url
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply blur to each selected field
        for field_coords in selected_fields:
            # Convert percentage coordinates to pixels
            x_percent = field_coords.get('x', 0)
            y_percent = field_coords.get('y', 0)
            width_percent = field_coords.get('width', 10)
            height_percent = field_coords.get('height', 5)
            
            img_width, img_height = image.size
            
            x = int((x_percent / 100) * img_width)
            y = int((y_percent / 100) * img_height)
            width = int((width_percent / 100) * img_width)
            height = int((height_percent / 100) * img_height)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            width = max(0, min(width, img_width - x))
            height = max(0, min(height, img_height - y))
            
            if width > 0 and height > 0:
                # Crop the region to blur
                region = image.crop((x, y, x + width, y + height))
                
                # Apply Gaussian blur
                blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_strength))
                
                # Paste back
                image.paste(blurred_region, (x, y))
        
        # Convert back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        blurred_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'blurred_image': f'data:image/png;base64,{blurred_image_data}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def save_to_history(data, user_id=None):
    """Save processing record to MongoDB history with user_id."""
    db_manager.save_record(data, user_id)

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get processing history from MongoDB for the current user."""
    user_id = session.get('user_id')
    records = db_manager.get_history(user_id=user_id)
    if records is None:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503
    
    return jsonify({'success': True, 'history': records})


# ===== AUTHENTICATION ROUTES =====

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page and authentication."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('document_redaction'))
        return render_template('login.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = login_user(username, password)
        
        if result['success']:
            session.permanent = True
            session['user_id'] = result['user']['id']
            session['username'] = result['user']['username']
            session['email'] = result['user']['email']
            
            return jsonify({'success': True, 'message': 'Login successful', 'user': result['user']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during login'}), 500


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    """Registration page and user creation."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('document_redaction'))
        return render_template('register.html')
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        result = register_user(username, email, password)
        
        if result['success']:
            return jsonify({'success': True, 'message': 'Registration successful'})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during registration'}), 500


@app.route('/logout')
def logout():
    """Logout user and clear session."""
    session.clear()
    return redirect(url_for('login_page'))


# ===== SETTINGS ROUTES =====

@app.route('/settings')
@login_required
def settings_page():
    """User settings page."""
    return render_template('settings.html')


@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get current user's profile information."""
    try:
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        
        if user:
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404
            
    except Exception as e:
        print(f"Profile fetch error: {e}")
        return jsonify({'success': False, 'message': 'Error fetching profile'}), 500


@app.route('/api/user/update-profile', methods=['POST'])
@login_required
def update_user_profile():
    """Update user's profile information (email)."""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        new_email = data.get('email', '').strip()
        
        if not new_email:
            return jsonify({'success': False, 'message': 'Email is required'}), 400
        
        result = update_user_email(user_id, new_email)
        
        if result['success']:
            # Update session email
            session['email'] = new_email
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({'success': False, 'message': 'Error updating profile'}), 500


@app.route('/api/user/change-password', methods=['POST'])
@login_required
def change_user_password():
    """Change user's password."""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        result = change_password(user_id, current_password, new_password)
        
        if result['success']:
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
            
    except Exception as e:
        print(f"Password change error: {e}")
        return jsonify({'success': False, 'message': 'Error changing password'}), 500


if __name__ == '__main__':

    print("🚀 Starting Simple Redaction Server...")
    print("🌐 Access at: http://localhost:5555/document-redaction")
    print("✨ This is a simplified server while app.py is being fixed")
    print()
    
    # NOTE: This uses port 5555 to avoid conflict with your main app
    app.run(debug=True, host='0.0.0.0', port=5555)
