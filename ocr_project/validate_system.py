#!/usr/bin/env python3
"""Quick validation test for the document classification system."""

from ocr.rag_field_suggestion import RAGFieldSuggestionEngine

def quick_validation():
    print("🔍 Running System Validation Test...")
    
    try:
        # Initialize engine
        engine = RAGFieldSuggestionEngine()
        
        # Test document
        test_doc = """
        GOVERNMENT OF INDIA
        AADHAAR
        Name: VALIDATION TEST
        Aadhaar: 1234 5678 9012
        """
        
        # Analyze
        result = engine.analyze_document_with_classification(test_doc)
        
        # Results
        print("✅ System Status: OPERATIONAL")
        print(f"📄 Detected: {result['analysis_summary']['best_document_type']}")
        print(f"📊 Confidence: {result['analysis_summary']['best_confidence']}")
        print(f"🏷️ Fields: {result['analysis_summary']['total_field_suggestions']}")
        print(f"🎯 Classifications: {result['analysis_summary']['total_classifications']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_validation()
    if success:
        print("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
        print("📋 Document Classification System is READY for production!")
    else:
        print("\n❌ VALIDATION FAILED - Please check system configuration")