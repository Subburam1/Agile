#!/usr/bin/env python3
"""
Test Field Extraction Pipeline API Compatibility
Tests that the fixed pipeline works correctly with optional parameters
"""
import sys
import os

def test_pipeline_compatibility():
    """Test that the field extraction pipeline accepts optional document_image_path"""
    print("🧪 Testing Field Extraction Pipeline API Compatibility")
    print("="*50)
    
    try:
        # Import the fixed pipeline
        print("1. 📥 Importing field extraction pipeline...")
        from field_extraction_pipeline_new import FieldExtractionPipeline
        print("   ✅ Import successful")
        
        # Initialize the pipeline
        print("2. 🔧 Initializing pipeline...")
        pipeline = FieldExtractionPipeline()
        print("   ✅ Pipeline initialized")
        
        # Test sample text
        sample_text = """
        Name: John Doe
        Email: john.doe@email.com
        Phone: +1-234-567-8900
        Address: 123 Main Street
        City: New York
        ZIP Code: 12345
        """
        
        # Test 1: Call without document_image_path (old API style)
        print("3. 🧪 Testing API call without document_image_path...")
        try:
            result1 = pipeline.extract_fields_from_text(sample_text)
            print(f"   ✅ Success! Returned type: {type(result1)}")
            print(f"   📊 Extracted fields: {len(result1.extracted_fields) if hasattr(result1, 'extracted_fields') else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 2: Call with document_image_path (new API style)
        print("4. 🧪 Testing API call with document_image_path...")
        try:
            result2 = pipeline.extract_fields_from_text(sample_text, document_image_path="test.png")
            print(f"   ✅ Success! Returned type: {type(result2)}")
            print(f"   📊 Extracted fields: {len(result2.extracted_fields) if hasattr(result2, 'extracted_fields') else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Verify return type structure
        print("5. 🔍 Verifying return structure...")
        if hasattr(result1, 'extracted_fields') and hasattr(result1, 'document_type'):
            print("   ✅ Return structure is correct (FieldAnalysisResult)")
        else:
            print("   ❌ Return structure is incorrect")
            return False
        
        print("\n🎉 API COMPATIBILITY TEST SUCCESSFUL!")
        print("✅ Fixed issues:")
        print("   - Method accepts optional document_image_path parameter")
        print("   - Returns structured FieldAnalysisResult objects")
        print("   - Backward compatible with old calls")
        print("   - Forward compatible with new calls")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_compatibility()
    exit(0 if success else 1)