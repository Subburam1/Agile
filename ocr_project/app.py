"""Flask web application for OCR text extraction with automatic preprocessing"""

import os
import tempfile
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import pytesseract
import logging
import atexit

from ocr import extract_text, preprocess_image, DocumentProcessor
from ocr.certificate_ocr import extract_certificate_text, preprocess_certificate
from ocr.mrz_ocr import extract_mrz_text, MRZParser, extract_raw_mrz_text
from ocr.rag_field_suggestion import RAGFieldSuggestionEngine
from ocr.deep_learning_ocr import extract_text_deep_learning, get_deep_learning_ocr_info
from document_history_db import db_manager, cleanup_db


# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


# Initialize document processor and RAG engine
doc_processor = DocumentProcessor()
rag_engine = RAGFieldSuggestionEngine()

def detect_document_type(image_path):
    """Detect document type using comprehensive detection system."""
    try:
        # Quick OCR scan to detect document type
        quick_text = extract_text(image_path, config='--psm 6')
        
        if not quick_text or len(quick_text.strip()) < 10:
            # Try different PSM modes if first attempt fails
            psm_modes = ['--psm 3', '--psm 1', '--psm 4', '--psm 8']
            for psm in psm_modes:
                try:
                    quick_text = extract_text(image_path, config=psm)
                    if quick_text and len(quick_text.strip()) >= 10:
                        break
                except Exception as e:
                    print(f"PSM {psm} failed: {e}")
                    continue
        
        print(f"📝 Extracted text for classification (length: {len(quick_text)}): {quick_text[:200]}...")
        
        # Use comprehensive document detection
        doc_type, confidence = doc_processor.detector.detect_document_type(quick_text)
        print(f"🎯 Document type detected: {doc_type} (confidence: {confidence:.2f})")
        
        return doc_type
            
    except Exception as e:
        print(f"❌ Document type detection failed: {e}")
        return 'general'


def extract_text_with_fallbacks(image_path):
    """Extract text using multiple OCR strategies for better reliability."""
    strategies = [
        {'config': '--psm 6', 'description': 'Single uniform block'},
        {'config': '--psm 3', 'description': 'Fully automatic page segmentation'},
        {'config': '--psm 4', 'description': 'Single column of text'},
        {'config': '--psm 8', 'description': 'Single word'},
        {'config': '--psm 1', 'description': 'Automatic page segmentation with OSD'},
        {'config': '--psm 7', 'description': 'Single text line'},
        {'config': '--psm 13', 'description': 'Raw line, treat as single text line'}
    ]
    
    best_text = ""
    best_length = 0
    
    for strategy in strategies:
        try:
            text = extract_text(image_path, config=strategy['config'])
            if text and len(text.strip()) > best_length:
                best_text = text
                best_length = len(text.strip())
                print(f"✅ OCR strategy '{strategy['description']}' produced {len(text)} characters")
            
            # If we got a reasonable amount of text, use it
            if len(text.strip()) > 50:
                print(f"🎯 Using OCR strategy: {strategy['description']}")
                return text
                
        except Exception as e:
            print(f"⚠️ OCR strategy '{strategy['description']}' failed: {e}")
            continue
    
    # Return the best result we found
    if best_text:
        print(f"📋 Using best OCR result with {len(best_text)} characters")
        return best_text
    else:
        print("❌ All OCR strategies failed")
        return "No text could be extracted from the image."


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Register cleanup function for database connection
atexit.register(cleanup_db)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_upload_folder():
    """Create upload folder if it doesn't exist."""
    upload_path = Path(app.config['UPLOAD_FOLDER'])
    upload_path.mkdir(exist_ok=True)

@app.route('/')
def index():
    """Main page with upload interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and OCR processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create upload folder if needed
        create_upload_folder()
        
        # Save uploaded file
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(file_path)

        # Get OCR engine preference from form
        ocr_engine = request.form.get('ocrEngine', 'auto')
        language = request.form.get('language', 'auto')
        preprocessing = request.form.get('preprocessing', 'auto')
        enable_classification = request.form.get('enableClassification', 'true').lower() == 'true'
        enable_field_suggestion = request.form.get('enableFieldSuggestion', 'true').lower() == 'true'
        
        try:
            # Detect document type
            doc_type = detect_document_type(str(file_path))
            
            if doc_type == 'mrz':
                # Process MRZ document (passport/ID)
                # extract_mrz_text already returns parsed data, not raw text
                mrz_data = extract_mrz_text(str(file_path))
                
                # Clean up uploaded file
                file_path.unlink()
                
                # Check if extraction was successful
                if 'error' in mrz_data:
                    return jsonify({
                        'success': False,
                        'error': mrz_data['error'],
                        'filename': filename,
                        'document_type': 'mrz'
                    })
                
                # Save to document history
                extracted_text = mrz_data.get('raw_ocr_text', 'MRZ data extracted')
                record_id = db_manager.save_document_record(
                    filename=filename,
                    document_type='mrz',
                    extracted_text=extracted_text,
                    confidence=0.95,  # MRZ typically has high confidence
                    processing_metadata={'method_used': 'MRZ OCR (passport/ID documents)'},
                    structured_data=mrz_data
                )
                
                return jsonify({
                    'success': True,
                    'extracted_text': extracted_text,
                    'filename': filename,
                    'preprocessing_applied': True,
                    'document_type': 'mrz',
                    'method_used': 'MRZ OCR (passport/ID documents)',
                    'structured_data': mrz_data,
                    'record_id': record_id
                })
                
            elif doc_type == 'certificate':
                # Use only raw OCR for certificates (works better)
                raw_text = extract_text(str(file_path))
                
                # Extract structured data from the raw text
                from ocr.certificate_ocr import extract_certificate_structure
                structured_data = extract_certificate_structure(raw_text)
                
                # Save to document history
                record_id = db_manager.save_document_record(
                    filename=filename,
                    document_type='certificate',
                    extracted_text=raw_text.strip(),
                    confidence=0.85,  # Certificates typically have good confidence
                    processing_metadata={'method_used': 'Raw OCR (best for certificates)'},
                    structured_data=structured_data
                )
                
                # Clean up uploaded file
                file_path.unlink()
                
                return jsonify({
                    'success': True,
                    'extracted_text': raw_text.strip(),
                    'filename': filename,
                    'preprocessing_applied': False,
                    'document_type': 'certificate',
                    'method_used': 'Raw OCR (best for certificates)',
                    'structured_data': structured_data,
                    'record_id': record_id
                })
            else:
                # Choose OCR method based on selection
                print(f"🔧 Using OCR engine: {ocr_engine}")
                
                if ocr_engine == 'deep_learning' or ocr_engine == 'auto':
                    # Try deep learning OCR
                    try:
                        # Use specified engine or auto detection
                        dl_engine = 'auto' if ocr_engine == 'auto' else 'easyocr'
                        print(f"🧠 Attempting deep learning OCR with engine: {dl_engine}")
                        dl_result = extract_text_deep_learning(str(file_path), engine=dl_engine)
                        
                        if 'error' not in dl_result and dl_result.get('text'):
                            raw_text = dl_result['text']
                            dl_confidence = dl_result.get('confidence', 0.0)
                            processing_method = f"Deep Learning OCR ({dl_result.get('engine', 'Neural Network')})"
                            dl_metadata = dl_result.get('metadata', {})
                            text_blocks = dl_result.get('text_blocks', [])
                            layout_analysis = dl_result.get('layout_analysis', {})
                            print(f"✅ Deep learning OCR successful. Text length: {len(raw_text)}, Confidence: {dl_confidence:.2f}")
                        else:
                            # Fallback to traditional OCR if deep learning fails
                            print(f"⚠️ Deep learning OCR failed: {dl_result.get('error', 'Unknown')}. Falling back to traditional OCR")
                            raw_text = extract_text_with_fallbacks(str(file_path))
                            dl_confidence = 0.7  # Default confidence for traditional OCR
                            processing_method = "Traditional OCR (Tesseract) - Deep Learning Fallback"
                            dl_metadata = {'fallback_reason': dl_result.get('error', 'Unknown')}
                            text_blocks = []
                            layout_analysis = {}
                    except Exception as e:
                        # Final fallback to traditional OCR
                        print(f"❌ Deep learning OCR exception: {e}. Falling back to traditional OCR")
                        raw_text = extract_text_with_fallbacks(str(file_path))
                        dl_confidence = 0.7
                        processing_method = "Traditional OCR (Tesseract) - Exception Fallback"
                        dl_metadata = {'fallback_reason': str(e)}
                        text_blocks = []
                        layout_analysis = {}
                elif ocr_engine == 'traditional':
                    # Use traditional OCR only
                    print("🔧 Using traditional OCR (Tesseract)")
                    raw_text = extract_text_with_fallbacks(str(file_path))
                    dl_confidence = 0.7
                    processing_method = "Traditional OCR (Tesseract)"
                    dl_metadata = {'engine_selection': 'traditional'}
                    text_blocks = []
                    layout_analysis = {}
                elif ocr_engine == 'benchmark':
                    # Compare all engines (simplified version)
                    print("⚖️ Running OCR benchmark comparison")
                    traditional_text = extract_text_with_fallbacks(str(file_path))
                    try:
                        dl_result = extract_text_deep_learning(str(file_path), engine='auto')
                        if 'error' not in dl_result and dl_result.get('text'):
                            # Use the better result based on confidence and text length
                            dl_confidence_score = dl_result.get('confidence', 0)
                            dl_text_length = len(dl_result['text'].strip())
                            traditional_text_length = len(traditional_text.strip())
                            
                            # Scoring logic: prefer longer text with higher confidence
                            dl_score = dl_confidence_score * 0.7 + (dl_text_length / max(dl_text_length, traditional_text_length, 1)) * 0.3
                            traditional_score = 0.7 * 0.7 + (traditional_text_length / max(dl_text_length, traditional_text_length, 1)) * 0.3
                            
                            print(f"📊 Benchmark scores - DL: {dl_score:.3f}, Traditional: {traditional_score:.3f}")
                            
                            if dl_score > traditional_score and dl_confidence_score > 0.5:
                                raw_text = dl_result['text']
                                dl_confidence = dl_result.get('confidence', 0.0)
                                processing_method = f"Benchmark - Deep Learning OCR ({dl_result.get('engine', 'Neural Network')})"
                                dl_metadata = dl_result.get('metadata', {})
                                dl_metadata['benchmark_traditional_length'] = len(traditional_text)
                                dl_metadata['benchmark_scores'] = {'dl': dl_score, 'traditional': traditional_score}
                                text_blocks = dl_result.get('text_blocks', [])
                                layout_analysis = dl_result.get('layout_analysis', {})
                            else:
                                raw_text = traditional_text
                                dl_confidence = 0.7
                                processing_method = "Benchmark - Traditional OCR (Tesseract)"
                                dl_metadata = {'benchmark_dl_confidence': dl_result.get('confidence', 0), 
                                             'benchmark_scores': {'dl': dl_score, 'traditional': traditional_score}}
                                text_blocks = []
                                layout_analysis = {}
                        else:
                            raw_text = traditional_text
                            dl_confidence = 0.7
                            processing_method = "Benchmark - Traditional OCR (Tesseract) - DL Failed"
                            dl_metadata = {'benchmark_dl_error': dl_result.get('error', 'Unknown')}
                            text_blocks = []
                            layout_analysis = {}
                    except Exception as e:
                        raw_text = traditional_text
                        dl_confidence = 0.7
                        processing_method = "Benchmark - Traditional OCR (Tesseract) - DL Exception"
                        dl_metadata = {'benchmark_dl_exception': str(e)}
                        text_blocks = []
                        layout_analysis = {}
                else:
                    # Default to auto mode
                    try:
                        dl_result = extract_text_deep_learning(str(file_path), engine='auto')
                        if 'error' not in dl_result and dl_result.get('text'):
                            raw_text = dl_result['text']
                            dl_confidence = dl_result.get('confidence', 0.0)
                            processing_method = f"Deep Learning OCR ({dl_result.get('engine', 'Neural Network')})"
                            dl_metadata = dl_result.get('metadata', {})
                            text_blocks = dl_result.get('text_blocks', [])
                            layout_analysis = dl_result.get('layout_analysis', {})
                        else:
                            # Fallback to traditional OCR
                            raw_text = extract_text(str(file_path))
                            dl_confidence = 0.7  # Default confidence for traditional OCR
                            processing_method = "Traditional OCR (Tesseract) - Deep Learning Fallback"
                            dl_metadata = {'fallback_reason': dl_result.get('error', 'Unknown')}
                            text_blocks = []
                            layout_analysis = {}
                    except Exception as e:
                        # Final fallback to traditional OCR
                        raw_text = extract_text(str(file_path))
                        dl_confidence = 0.7
                        processing_method = "Traditional OCR (Tesseract) - Exception Fallback"
                        dl_metadata = {'fallback_reason': str(e)}
                        text_blocks = []
                        layout_analysis = {}
                
                # Process with comprehensive document processor
                processed_data = doc_processor.process_document(str(file_path), raw_text)
                
                # Override confidence with deep learning confidence if available
                if dl_confidence > processed_data.get('confidence', 0):
                    processed_data['confidence'] = dl_confidence
                
                # Generate enhanced RAG analysis with document classification
                try:
                    rag_analysis = rag_engine.analyze_document_with_classification(
                        raw_text, 
                        top_k=8
                    )
                except Exception as e:
                    print(f"RAG processing error: {e}")
                    rag_analysis = {
                        "document_classifications": [],
                        "field_suggestions": [],
                        "analysis_summary": {
                            "total_classifications": 0,
                            "best_document_type": "UNKNOWN",
                            "best_confidence": "0.000",
                            "total_field_suggestions": 0,
                            "high_confidence_fields": 0
                        }
                    }
                
                # Clean up uploaded file
                file_path.unlink()
                
                # Get document-specific method description
                method_map = {
                    'invoice': 'Invoice OCR (structured data extraction)',
                    'receipt': 'Receipt OCR (transaction data extraction)',
                    'form': 'Form OCR (field data extraction)',
                    'business_card': 'Business Card OCR (contact extraction)',
                    'medical': 'Medical Document OCR (patient data)',
                    'legal': 'Legal Document OCR (contract analysis)',
                    'academic': 'Academic Document OCR (transcript data)',
                    'financial': 'Financial Document OCR (account data)',
                    'government': 'Government Document OCR (official data)',
                    'general': 'General OCR (text extraction)'
                }
                
                # Save to document history
                record_id = db_manager.save_document_record(
                    filename=filename,
                    document_type=processed_data['document_type'],
                    extracted_text=raw_text.strip(),
                    confidence=processed_data.get('confidence', 0.0),
                    processing_metadata={
                        'method_used': processing_method,
                        'processed_at': processed_data.get('processed_at'),
                        'confidence_score': f"{processed_data.get('confidence', 0.0):.2f}",
                        'rag_enabled': True,
                        'suggestions_count': rag_analysis['analysis_summary']['total_field_suggestions'],
                        'document_classification': {
                            'best_type': rag_analysis['analysis_summary']['best_document_type'],
                            'confidence': rag_analysis['analysis_summary']['best_confidence'],
                            'total_classifications': rag_analysis['analysis_summary']['total_classifications'],
                            'high_confidence_types': rag_analysis['analysis_summary'].get('high_confidence_doc_types', 0)
                        },
                        'deep_learning_metadata': dl_metadata,
                        'text_blocks_count': len(text_blocks),
                        'layout_analysis': layout_analysis
                    },
                    structured_data=processed_data.get('structured_data', {}),
                    rag_suggestions=rag_analysis['field_suggestions'],
                    document_classifications=rag_analysis['document_classifications']
                )
                
                return jsonify({
                    'success': True,
                    'extracted_text': raw_text.strip(),
                    'filename': filename,
                    'preprocessing_applied': False,
                    'document_type': processed_data['document_type'],
                    'confidence': processed_data.get('confidence', 0.0),
                    'method_used': processing_method,
                    'structured_data': processed_data.get('structured_data', {}),
                    'document_classifications': rag_analysis['document_classifications'],
                    'rag_suggestions': rag_analysis['field_suggestions'],
                    'deep_learning_info': {
                        'engine': dl_metadata.get('engine', 'Traditional'),
                        'text_blocks': len(text_blocks),
                        'layout_structure': layout_analysis.get('structure', 'unknown'),
                        'confidence_distribution': layout_analysis.get('confidence_distribution', {}),
                        'regions': layout_analysis.get('regions', [])
                    },
                    'processing_metadata': {
                        'processed_at': processed_data.get('processed_at'),
                        'confidence_score': f"{processed_data.get('confidence', 0.0):.2f}",
                        'rag_enabled': True,
                        'suggestions_count': rag_analysis['analysis_summary']['total_field_suggestions'],
                        'document_classification': {
                            'best_type': rag_analysis['analysis_summary']['best_document_type'],
                            'confidence': rag_analysis['analysis_summary']['best_confidence'],
                            'total_classifications': rag_analysis['analysis_summary']['total_classifications'],
                            'high_confidence_types': rag_analysis['analysis_summary'].get('high_confidence_doc_types', 0)
                        },
                        'deep_learning_enabled': True,
                        'high_confidence_filtering': True
                    },
                    'record_id': record_id
                })
            
        except Exception as e:
            # Clean up uploaded file on error
            if file_path.exists():
                file_path.unlink()
            return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Test if Tesseract is working
        from ocr.ocr import get_available_languages
        languages = get_available_languages()
        return jsonify({
            'status': 'healthy',
            'tesseract_available': True,
            'available_languages': languages
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'tesseract_available': False,
            'error': str(e)
        }), 500

@app.route('/rag/suggest', methods=['POST'])
def rag_suggest_fields():
    """API endpoint for RAG field suggestions."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text parameter is required'}), 400
        
        text = data['text']
        document_type = data.get('document_type')
        top_k = data.get('top_k', 8)
        
        # Generate suggestions
        suggestions = rag_engine.suggest_fields(text, document_type, top_k)
        summary = rag_engine.get_field_suggestions_summary(suggestions)
        
        return jsonify({
            'success': True,
            'suggestions': summary,
            'metadata': {
                'text_length': len(text),
                'document_type': document_type,
                'top_k': top_k
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'RAG suggestion failed: {str(e)}'}), 500

@app.route('/rag/knowledge-base')
def rag_knowledge_base():
    """Get information about RAG knowledge base."""
    try:
        patterns = rag_engine.knowledge_base.field_patterns
        
        # Group patterns by document type
        doc_types = {}
        for pattern in patterns:
            if pattern.document_type not in doc_types:
                doc_types[pattern.document_type] = []
            doc_types[pattern.document_type].append({
                'field_name': pattern.field_name,
                'field_type': pattern.field_type,
                'description': pattern.description,
                'keywords': pattern.keywords[:5],  # Limit for display
                'examples': pattern.examples[:3]   # Limit for display
            })
        
        return jsonify({
            'total_patterns': len(patterns),
            'document_types': list(doc_types.keys()),
            'patterns_by_type': doc_types
        })
        
    except Exception as e:
        return jsonify({'error': f'Knowledge base access failed: {str(e)}'}), 500

# Deep Learning OCR API Endpoints

@app.route('/api/deep-learning-ocr/info')
def deep_learning_ocr_info():
    """Get information about available deep learning OCR engines."""
    try:
        info = get_deep_learning_ocr_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get deep learning OCR info: {str(e)}'}), 500

@app.route('/api/deep-learning-ocr/process', methods=['POST'])
def process_with_deep_learning_ocr():
    """Process uploaded file with specific deep learning OCR engine."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get engine preference from form data
        engine = request.form.get('engine', 'auto')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create upload folder if needed
        create_upload_folder()
        
        # Save uploaded file
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(file_path)
        
        try:
            # Process with deep learning OCR
            result = extract_text_deep_learning(
                str(file_path),
                engine=engine,
                confidence_threshold=confidence_threshold
            )
            
            # Clean up uploaded file
            file_path.unlink()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error'],
                    'filename': filename
                })
            
            return jsonify({
                'success': True,
                'extracted_text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'engine_used': result.get('engine', 'Unknown'),
                'text_blocks': result.get('text_blocks', []),
                'layout_analysis': result.get('layout_analysis', {}),
                'processing_time': result.get('processing_time', 0),
                'metadata': result.get('metadata', {}),
                'filename': filename,
                'bounding_boxes': result.get('bounding_boxes', [])
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if file_path.exists():
                file_path.unlink()
            return jsonify({'error': f'Deep learning OCR processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/deep-learning-ocr/benchmark', methods=['POST'])
def benchmark_deep_learning_ocr():
    """Benchmark different deep learning OCR engines on the same image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Create upload folder if needed
        create_upload_folder()
        
        # Save uploaded file
        file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(file_path)
        
        try:
            from ocr.deep_learning_ocr import dl_ocr
            
            # Benchmark all available engines
            benchmark_results = dl_ocr.benchmark_engines(str(file_path))
            
            # Clean up uploaded file
            file_path.unlink()
            
            return jsonify({
                'success': True,
                'benchmark_results': benchmark_results,
                'filename': filename,
                'available_engines': dl_ocr.get_available_engines()
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if file_path.exists():
                file_path.unlink()
            return jsonify({'error': f'Benchmarking failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# Document History API Endpoints

@app.route('/api/document-history')
def get_document_history_legacy():
    """Legacy endpoint for document history - redirects to /api/history."""
    return get_document_history()

@app.route('/api/history')
def get_document_history():
    """Get document processing history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        document_type = request.args.get('document_type')
        days_back = request.args.get('days_back', 30, type=int)
        
        # Validate parameters
        if limit > 100:
            limit = 100
        if days_back > 365:
            days_back = 365
            
        records = db_manager.get_document_history(
            limit=limit,
            document_type=document_type,
            days_back=days_back
        )
        
        return jsonify({
            'success': True,
            'records': records,
            'count': len(records),
            'filters': {
                'limit': limit,
                'document_type': document_type,
                'days_back': days_back
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve document history: {str(e)}'}), 500

@app.route('/api/history/<record_id>')
def get_document_by_id(record_id):
    """Get specific document record by ID."""
    try:
        record = db_manager.get_document_by_id(record_id)
        
        if record:
            return jsonify({
                'success': True,
                'record': record
            })
        else:
            return jsonify({'error': 'Document record not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve document: {str(e)}'}), 500

@app.route('/api/history/statistics')
def get_history_statistics():
    """Get document processing statistics."""
    try:
        stats = db_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve statistics: {str(e)}'}), 500

@app.route('/api/history/cleanup', methods=['POST'])
def cleanup_old_records():
    """Delete old document records."""
    try:
        data = request.get_json()
        days_old = data.get('days_old', 90) if data else 90
        
        # Validate parameter
        if days_old < 1:
            return jsonify({'error': 'days_old must be at least 1'}), 400
        if days_old > 3650:  # 10 years max
            days_old = 3650
            
        deleted_count = db_manager.delete_old_records(days_old)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'days_old': days_old
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to cleanup records: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.route('/api/diagnostics/ocr')
def ocr_diagnostics():
    """Provide OCR system diagnostics and health check."""
    try:
        diagnostics = {
            'tesseract_available': True,
            'tesseract_path': None,
            'tesseract_version': None,
            'deep_learning_ocr': get_deep_learning_ocr_info(),
            'supported_languages': [],
            'test_results': {}
        }
        
        # Test Tesseract availability
        try:
            import pytesseract
            diagnostics['tesseract_path'] = pytesseract.pytesseract.tesseract_cmd
            diagnostics['tesseract_version'] = str(pytesseract.get_tesseract_version())
            diagnostics['supported_languages'] = pytesseract.get_languages()
        except Exception as e:
            diagnostics['tesseract_available'] = False
            diagnostics['tesseract_error'] = str(e)
        
        # Test simple OCR functionality
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple test image with text
            test_image = Image.new('RGB', (200, 50), color='white')
            draw = ImageDraw.Draw(test_image)
            draw.text((10, 10), "TEST OCR", fill='black')
            
            # Save temporarily and test OCR
            create_upload_folder()
            test_path = Path(app.config['UPLOAD_FOLDER']) / 'test_ocr.png'
            test_image.save(test_path)
            
            # Test traditional OCR
            traditional_result = extract_text(str(test_path))
            diagnostics['test_results']['traditional_ocr'] = {
                'success': 'TEST' in traditional_result.upper(),
                'result': traditional_result[:100],
                'length': len(traditional_result)
            }
            
            # Test deep learning OCR if available
            if diagnostics['deep_learning_ocr']['easyocr_available'] or diagnostics['deep_learning_ocr']['paddleocr_available']:
                dl_result = extract_text_deep_learning(str(test_path))
                diagnostics['test_results']['deep_learning_ocr'] = {
                    'success': 'error' not in dl_result,
                    'result': dl_result.get('text', '')[:100] if 'error' not in dl_result else dl_result.get('error', ''),
                    'confidence': dl_result.get('confidence', 0),
                    'engine': dl_result.get('engine', 'unknown')
                }
            
            # Clean up test file
            if test_path.exists():
                test_path.unlink()
                
        except Exception as e:
            diagnostics['test_results']['error'] = str(e)
        
        return jsonify({
            'success': True,
            'diagnostics': diagnostics,
            'recommendations': generate_ocr_recommendations(diagnostics)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to run OCR diagnostics: {str(e)}'}), 500

def generate_ocr_recommendations(diagnostics):
    """Generate recommendations based on OCR diagnostics."""
    recommendations = []
    
    if not diagnostics['tesseract_available']:
        recommendations.append({
            'severity': 'critical',
            'issue': 'Tesseract OCR not available',
            'solution': 'Install Tesseract OCR and ensure it is in your system PATH'
        })
    
    if not diagnostics['deep_learning_ocr']['easyocr_available'] and not diagnostics['deep_learning_ocr']['paddleocr_available']:
        recommendations.append({
            'severity': 'warning', 
            'issue': 'No deep learning OCR engines available',
            'solution': 'Install EasyOCR (pip install easyocr) or PaddleOCR (pip install paddleocr) for better accuracy'
        })
    
    test_results = diagnostics.get('test_results', {})
    if 'traditional_ocr' in test_results and not test_results['traditional_ocr']['success']:
        recommendations.append({
            'severity': 'error',
            'issue': 'Traditional OCR test failed',
            'solution': 'Check Tesseract installation and configuration'
        })
    
    if 'deep_learning_ocr' in test_results and not test_results['deep_learning_ocr']['success']:
        recommendations.append({
            'severity': 'warning',
            'issue': 'Deep learning OCR test failed', 
            'solution': 'Check EasyOCR/PaddleOCR installation and dependencies'
        })
    
    if len(recommendations) == 0:
        recommendations.append({
            'severity': 'info',
            'issue': 'All OCR systems working correctly',
            'solution': 'No action needed - OCR system is healthy'
        })
    
    return recommendations

if __name__ == '__main__':
    # Create upload folder on startup
    create_upload_folder()
    
    print("🚀 Starting Raw OCR Web Application...")
    print("📁 Upload folder:", Path(app.config['UPLOAD_FOLDER']).absolute())
    print("🌐 Access the application at: http://localhost:5000")
    print("✨ Features: Raw OCR extraction, certificate detection, drag-and-drop upload")
    
    app.run(debug=True, host='0.0.0.0', port=5000)