#!/usr/bin/env python3
"""Debug test to understand document detection scoring"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def debug_detection():
    """Debug document detection to understand scoring."""
    print("🔍 Debug: Document Detection Scoring")
    print("=" * 50)
    
    from ocr.document_types import DocumentTypeDetector
    
    detector = DocumentTypeDetector()
    
    # Test invoice detection
    invoice_text = "INVOICE #12345\nBILL TO: John Doe\nTOTAL: $123.45"
    print(f"\nTest Text: {invoice_text}")
    print("-" * 30)
    
    text_upper = invoice_text.upper()
    for doc_type, config in detector.document_patterns.items():
        score = detector._calculate_score(text_upper, config)
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_upper)
        pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, text_upper))
        
        print(f"{doc_type}:")
        print(f"  Keywords found: {keyword_matches}/{len(config['keywords'])}")
        print(f"  Patterns found: {pattern_matches}/{len(config['patterns'])}")
        print(f"  Score: {score:.3f} (threshold: {config['confidence_threshold']})")
        print(f"  Passes: {'✅' if score >= config['confidence_threshold'] else '❌'}")
        
        # Show which keywords matched
        matched_keywords = [kw for kw in config['keywords'] if kw in text_upper]
        if matched_keywords:
            print(f"  Matched keywords: {matched_keywords}")
        print()

if __name__ == '__main__':
    import re  # Need to import re for the debug function
    debug_detection()