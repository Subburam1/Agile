"""
Advanced document type detection and processing for OCR system.
Supports multiple document types with specialized extraction logic.
"""

import re
from typing import Dict, List, Tuple, Any
from datetime import datetime


class DocumentTypeDetector:
    """Advanced document type detection with confidence scoring."""
    
    def __init__(self):
        self.document_patterns = {
            'invoice': {
                'keywords': [
                    'INVOICE', 'BILL TO', 'SHIP TO', 'SUBTOTAL', 'TAX', 'TOTAL',
                    'AMOUNT DUE', 'INVOICE NUMBER', 'DATE DUE', 'PAYMENT TERMS',
                    'QTY', 'QUANTITY', 'PRICE', 'DESCRIPTION', 'ITEM'
                ],
                'patterns': [
                    r'INVOICE\s*#?\s*\d+',
                    r'TOTAL\s*\$?\s*[\d,]+\.?\d*',
                    r'DUE\s*DATE',
                    r'BILL\s*TO:'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'receipt': {
                'keywords': [
                    'RECEIPT', 'THANK YOU', 'CHANGE', 'CASH', 'CREDIT',
                    'CARD', 'SUBTOTAL', 'TAX', 'TOTAL', 'STORE', 'LOCATION',
                    'TRANSACTION', 'TIME', 'DATE'
                ],
                'patterns': [
                    r'\$\s*\d+\.\d{2}',
                    r'TOTAL\s*\$?\s*\d+\.\d{2}',
                    r'CHANGE\s*\$?\s*\d+\.\d{2}',
                    r'\d{2}/\d{2}/\d{4}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.3
            },
            'form': {
                'keywords': [
                    'APPLICATION', 'FORM', 'NAME:', 'ADDRESS:', 'PHONE:',
                    'EMAIL:', 'SIGNATURE', 'DATE:', 'PLEASE PRINT',
                    'FIRST NAME', 'LAST NAME', 'ZIP CODE', 'STATE'
                ],
                'patterns': [
                    r'NAME:\s*[_\s]*',
                    r'ADDRESS:\s*[_\s]*',
                    r'PHONE:\s*[_\s]*',
                    r'EMAIL:\s*[_\s]*'
                ],
                'min_keywords': 2,
                'confidence_threshold': 0.5
            },
            'business_card': {
                'keywords': [
                    'CEO', 'MANAGER', 'DIRECTOR', 'PRESIDENT', 'VP',
                    'EMAIL', 'PHONE', 'MOBILE', 'OFFICE', 'FAX',
                    'WWW', 'HTTP', 'COM', 'ORG', 'NET'
                ],
                'patterns': [
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    r'\(\d{3}\)\s*\d{3}-\d{4}',
                    r'\d{3}-\d{3}-\d{4}',
                    r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'medical': {
                'keywords': [
                    'PATIENT', 'DOCTOR', 'PHYSICIAN', 'CLINIC', 'HOSPITAL',
                    'PRESCRIPTION', 'DIAGNOSIS', 'TREATMENT', 'MEDICAL',
                    'CHART', 'DOB', 'ALLERGIES', 'MEDICATION'
                ],
                'patterns': [
                    r'DOB:\s*\d{2}/\d{2}/\d{4}',
                    r'PATIENT\s*ID:\s*\d+',
                    r'DR\.\s*[A-Z][a-z]+',
                    r'RX\s*#?\d+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'legal': {
                'keywords': [
                    'COURT', 'LEGAL', 'CONTRACT', 'AGREEMENT', 'WHEREAS',
                    'THEREFORE', 'PARTY', 'WITNESS', 'NOTARY', 'SWORN',
                    'AFFIDAVIT', 'PLAINTIFF', 'DEFENDANT', 'ATTORNEY'
                ],
                'patterns': [
                    r'WHEREAS\s+',
                    r'THEREFORE\s+',
                    r'IN\s+WITNESS\s+WHEREOF',
                    r'STATE\s+OF\s+[A-Z]+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'academic': {
                'keywords': [
                    'UNIVERSITY', 'COLLEGE', 'SCHOOL', 'STUDENT', 'GRADE',
                    'TRANSCRIPT', 'COURSE', 'CREDIT', 'GPA', 'SEMESTER',
                    'DEGREE', 'BACHELOR', 'MASTER', 'GRADUATION'
                ],
                'patterns': [
                    r'GPA:\s*\d+\.\d+',
                    r'STUDENT\s*ID:\s*\d+',
                    r'CREDIT\s*HOURS?:\s*\d+',
                    r'SEMESTER:\s*[A-Z]+'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'financial': {
                'keywords': [
                    'BANK', 'STATEMENT', 'ACCOUNT', 'BALANCE', 'DEPOSIT',
                    'WITHDRAWAL', 'TRANSACTION', 'ROUTING', 'CHECK',
                    'PAYMENT', 'INTEREST', 'FEE', 'CREDIT', 'DEBIT'
                ],
                'patterns': [
                    r'ACCOUNT\s*#?\s*\d+',
                    r'BALANCE:\s*\$?\s*[\d,]+\.\d{2}',
                    r'ROUTING\s*#?\s*\d{9}',
                    r'\d{2}/\d{2}/\d{4}\s+[\d,]+\.\d{2}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'government': {
                'keywords': [
                    'GOVERNMENT', 'DEPARTMENT', 'BUREAU', 'AGENCY', 'FEDERAL',
                    'STATE', 'COUNTY', 'MUNICIPAL', 'LICENSE', 'PERMIT',
                    'REGISTRATION', 'OFFICIAL', 'SEAL', 'AUTHORIZED'
                ],
                'patterns': [
                    r'LICENSE\s*#?\s*[A-Z0-9]+',
                    r'PERMIT\s*#?\s*\d+',
                    r'REGISTRATION\s*#?\s*[A-Z0-9]+',
                    r'EXPIRES?:\s*\d{2}/\d{2}/\d{4}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            },
            'mrz': {
                'keywords': ['P<', 'I<', 'V<', '<<<', '<<'],
                'patterns': [
                    r'P<[A-Z]{3}[A-Z<]+',
                    r'I<[A-Z]{3}[A-Z<]+',
                    r'V<[A-Z]{3}[A-Z<]+',
                    r'[A-Z0-9<]{30,}'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.8
            },
            'certificate': {
                'keywords': [
                    'CERTIFICATE', 'APPRECIATION', 'ACHIEVEMENT', 'AWARD', 'RECOGNITION',
                    'PRESENTED TO', 'HEREBY CERTIFY', 'COMPLETION', 'DIPLOMA', 'HONOR'
                ],
                'patterns': [
                    r'CERTIFICATE\s+OF\s+[A-Z]+',
                    r'PRESENTED\s+TO\s*:?',
                    r'THIS\s+CERTIFIES\s+THAT',
                    r'IN\s+RECOGNITION\s+OF'
                ],
                'min_keywords': 1,
                'confidence_threshold': 0.5
            }
        }
    
    def detect_document_type(self, text: str, min_confidence: float = 0.7) -> Tuple[str, float]:
        """
        Detect document type with confidence score.
        
        Args:
            text: OCR extracted text
            min_confidence: Minimum confidence threshold (default 0.7 for high confidence)
            
        Returns:
            Tuple of (document_type, confidence_score)
        """
        text_upper = text.upper()
        scores = {}
        
        for doc_type, config in self.document_patterns.items():
            score = self._calculate_score(text_upper, config)
            if score >= config['confidence_threshold']:
                scores[doc_type] = score
        
        # Filter for high confidence results
        high_confidence_scores = {k: v for k, v in scores.items() if v >= min_confidence}
        
        if high_confidence_scores:
            best_type = max(high_confidence_scores.keys(), key=lambda k: high_confidence_scores[k])
            return best_type, high_confidence_scores[best_type]
        elif scores:
            # If no high confidence match but there are some matches, return the best one
            # but only if it meets the document type's own threshold
            best_type = max(scores.keys(), key=lambda k: scores[k])
            return best_type, scores[best_type]
        else:
            return 'general', 0.5
    
    def get_high_confidence_document_types(self, text: str, min_confidence: float = 0.7) -> List[Tuple[str, float]]:
        """
        Get all document types that meet the high confidence threshold.
        
        Args:
            text: OCR extracted text
            min_confidence: Minimum confidence threshold (default 0.7)
            
        Returns:
            List of (document_type, confidence_score) tuples sorted by confidence
        """
        text_upper = text.upper()
        high_confidence_types = []
        
        for doc_type, config in self.document_patterns.items():
            score = self._calculate_score(text_upper, config)
            if score >= max(config['confidence_threshold'], min_confidence):
                high_confidence_types.append((doc_type, score))
        
        # Sort by confidence score (highest first)
        high_confidence_types.sort(key=lambda x: x[1], reverse=True)
        return high_confidence_types
    
    def _calculate_score(self, text: str, config: Dict) -> float:
        """Calculate confidence score for a document type."""
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text)
        pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, text))
        
        # More generous scoring: give more weight to keyword matches
        # Keyword score (0-0.8) - increased from 0.7
        keyword_score = min(keyword_matches / len(config['keywords']) * 0.8, 0.8)
        
        # Pattern score (0-0.2) - decreased from 0.3
        pattern_score = min(pattern_matches / len(config['patterns']) * 0.2, 0.2) if config['patterns'] else 0
        
        total_score = keyword_score + pattern_score
        
        # More generous minimum keyword requirement boost
        if keyword_matches >= config['min_keywords']:
            # Bonus for meeting minimum requirements
            total_score += 0.2
            
        # Cap at 1.0
        return min(total_score, 1.0)


class DocumentProcessor:
    """Process different document types with specialized extraction."""
    
    def __init__(self):
        self.detector = DocumentTypeDetector()
    
    def process_document(self, image_path: str, raw_text: str) -> Dict[str, Any]:
        """
        Process document based on its detected type.
        
        Args:
            image_path: Path to the image file
            raw_text: Raw OCR text
            
        Returns:
            Dictionary with processed document data
        """
        doc_type, confidence = self.detector.detect_document_type(raw_text)
        
        result = {
            'document_type': doc_type,
            'confidence': confidence,
            'raw_text': raw_text,
            'processed_at': datetime.now().isoformat()
        }
        
        # Apply specialized processing based on document type
        if doc_type == 'invoice':
            result['structured_data'] = self._process_invoice(raw_text)
        elif doc_type == 'receipt':
            result['structured_data'] = self._process_receipt(raw_text)
        elif doc_type == 'form':
            result['structured_data'] = self._process_form(raw_text)
        elif doc_type == 'business_card':
            result['structured_data'] = self._process_business_card(raw_text)
        elif doc_type == 'medical':
            result['structured_data'] = self._process_medical(raw_text)
        elif doc_type == 'legal':
            result['structured_data'] = self._process_legal(raw_text)
        elif doc_type == 'academic':
            result['structured_data'] = self._process_academic(raw_text)
        elif doc_type == 'financial':
            result['structured_data'] = self._process_financial(raw_text)
        elif doc_type == 'government':
            result['structured_data'] = self._process_government(raw_text)
        elif doc_type == 'mrz':
            # MRZ processing is handled separately
            result['structured_data'] = {'note': 'MRZ processing handled by specialized module'}
        elif doc_type == 'certificate':
            # Certificate processing is handled separately
            result['structured_data'] = {'note': 'Certificate processing handled by specialized module'}
        else:
            result['structured_data'] = self._process_general(raw_text)
        
        return result
    
    def _process_invoice(self, text: str) -> Dict[str, Any]:
        """Extract structured data from invoice."""
        data = {}
        
        # Invoice number
        invoice_match = re.search(r'INVOICE\s*#?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if invoice_match:
            data['invoice_number'] = invoice_match.group(1)
        
        # Total amount
        total_match = re.search(r'TOTAL\s*:?\s*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
        if total_match:
            data['total_amount'] = total_match.group(1)
        
        # Date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            data['date'] = date_match.group(1)
        
        # Company/vendor (usually at the top)
        lines = text.split('\n')
        if lines:
            data['vendor'] = lines[0].strip()
        
        return data
    
    def _process_receipt(self, text: str) -> Dict[str, Any]:
        """Extract structured data from receipt."""
        data = {}
        
        # Store name (usually first line)
        lines = text.split('\n')
        if lines:
            data['store'] = lines[0].strip()
        
        # Total amount
        total_match = re.search(r'TOTAL\s*\$?\s*(\d+\.\d{2})', text, re.IGNORECASE)
        if total_match:
            data['total'] = total_match.group(1)
        
        # Date and time
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            data['date'] = date_match.group(1)
        
        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)', text, re.IGNORECASE)
        if time_match:
            data['time'] = time_match.group(1)
        
        # Items (basic extraction)
        items = re.findall(r'([A-Z][A-Za-z\s]+)\s+\$?(\d+\.\d{2})', text)
        if items:
            data['items'] = [{'name': item[0].strip(), 'price': item[1]} for item in items]
        
        return data
    
    def _process_form(self, text: str) -> Dict[str, Any]:
        """Extract structured data from form."""
        data = {}
        
        # Extract field-value pairs
        patterns = {
            'name': r'(?:FULL\s+)?NAME\s*:?\s*([A-Za-z\s]+)',
            'first_name': r'FIRST\s+NAME\s*:?\s*([A-Za-z]+)',
            'last_name': r'LAST\s+NAME\s*:?\s*([A-Za-z]+)',
            'address': r'ADDRESS\s*:?\s*([A-Za-z0-9\s,]+)',
            'phone': r'PHONE\s*:?\s*([\d\-\(\)\s]+)',
            'email': r'EMAIL\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'date': r'DATE\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[field] = match.group(1).strip()
        
        return data
    
    def _process_business_card(self, text: str) -> Dict[str, Any]:
        """Extract structured data from business card."""
        data = {}
        
        # Email
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
        if email_match:
            data['email'] = email_match.group(1)
        
        # Phone
        phone_match = re.search(r'(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})', text)
        if phone_match:
            data['phone'] = phone_match.group(1)
        
        # Website
        website_match = re.search(r'((?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text, re.IGNORECASE)
        if website_match:
            data['website'] = website_match.group(1)
        
        # Company (usually the largest/first line)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            data['company'] = lines[0]
        
        # Title/Position
        titles = ['CEO', 'MANAGER', 'DIRECTOR', 'PRESIDENT', 'VP', 'VICE PRESIDENT']
        for line in lines:
            if any(title in line.upper() for title in titles):
                data['title'] = line
                break
        
        return data
    
    def _process_medical(self, text: str) -> Dict[str, Any]:
        """Extract structured data from medical document."""
        data = {}
        
        # Patient name
        patient_match = re.search(r'PATIENT\s*:?\s*([A-Za-z\s,]+)', text, re.IGNORECASE)
        if patient_match:
            data['patient_name'] = patient_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DOB\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})', text, re.IGNORECASE)
        if dob_match:
            data['date_of_birth'] = dob_match.group(1)
        
        # Doctor name
        doctor_match = re.search(r'DR\.\s*([A-Za-z\s]+)', text, re.IGNORECASE)
        if doctor_match:
            data['doctor'] = doctor_match.group(1).strip()
        
        # Date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            data['date'] = date_match.group(1)
        
        return data
    
    def _process_legal(self, text: str) -> Dict[str, Any]:
        """Extract structured data from legal document."""
        data = {}
        
        # Document type
        if 'CONTRACT' in text.upper():
            data['document_type'] = 'Contract'
        elif 'AGREEMENT' in text.upper():
            data['document_type'] = 'Agreement'
        elif 'AFFIDAVIT' in text.upper():
            data['document_type'] = 'Affidavit'
        
        # Parties
        parties = re.findall(r'PARTY\s+[A-Z]\s*:?\s*([A-Za-z\s,]+)', text, re.IGNORECASE)
        if parties:
            data['parties'] = [party.strip() for party in parties]
        
        # Date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            data['date'] = date_match.group(1)
        
        return data
    
    def _process_academic(self, text: str) -> Dict[str, Any]:
        """Extract structured data from academic document."""
        data = {}
        
        # Student name
        student_match = re.search(r'STUDENT\s*:?\s*([A-Za-z\s,]+)', text, re.IGNORECASE)
        if student_match:
            data['student_name'] = student_match.group(1).strip()
        
        # GPA
        gpa_match = re.search(r'GPA\s*:?\s*(\d+\.\d+)', text, re.IGNORECASE)
        if gpa_match:
            data['gpa'] = gpa_match.group(1)
        
        # Degree
        degree_match = re.search(r'(BACHELOR|MASTER|DOCTORATE)\s+OF\s+([A-Za-z\s]+)', text, re.IGNORECASE)
        if degree_match:
            data['degree'] = f"{degree_match.group(1)} of {degree_match.group(2)}"
        
        # Institution
        lines = text.split('\n')
        for line in lines:
            if any(word in line.upper() for word in ['UNIVERSITY', 'COLLEGE', 'SCHOOL']):
                data['institution'] = line.strip()
                break
        
        return data
    
    def _process_financial(self, text: str) -> Dict[str, Any]:
        """Extract structured data from financial document."""
        data = {}
        
        # Account number
        account_match = re.search(r'ACCOUNT\s*#?\s*:?\s*(\d+)', text, re.IGNORECASE)
        if account_match:
            data['account_number'] = account_match.group(1)
        
        # Balance
        balance_match = re.search(r'BALANCE\s*:?\s*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if balance_match:
            data['balance'] = balance_match.group(1)
        
        # Bank name
        lines = text.split('\n')
        for line in lines:
            if 'BANK' in line.upper():
                data['bank'] = line.strip()
                break
        
        return data
    
    def _process_government(self, text: str) -> Dict[str, Any]:
        """Extract structured data from government document."""
        data = {}
        
        # License/Permit number
        license_match = re.search(r'(?:LICENSE|PERMIT)\s*#?\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
        if license_match:
            data['license_number'] = license_match.group(1)
        
        # Expiration date
        expires_match = re.search(r'EXPIRES?\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})', text, re.IGNORECASE)
        if expires_match:
            data['expiration_date'] = expires_match.group(1)
        
        # Issuing authority
        lines = text.split('\n')
        for line in lines:
            if any(word in line.upper() for word in ['DEPARTMENT', 'BUREAU', 'AGENCY']):
                data['issuing_authority'] = line.strip()
                break
        
        return data
    
    def _process_general(self, text: str) -> Dict[str, Any]:
        """Process general document with basic information extraction."""
        data = {}
        
        # Basic statistics
        lines = text.split('\n')
        data['line_count'] = len([line for line in lines if line.strip()])
        data['word_count'] = len(text.split())
        data['char_count'] = len(text)
        
        # Extract dates
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
        if dates:
            data['dates_found'] = dates
        
        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if emails:
            data['emails_found'] = emails
        
        # Extract phone numbers
        phones = re.findall(r'(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})', text)
        if phones:
            data['phones_found'] = phones
        
        return data