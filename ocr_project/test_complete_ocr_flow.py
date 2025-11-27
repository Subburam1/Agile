#!/usr/bin/env python3
"""
Test Complete OCR Flow Implementation
Tests the complete sequential processing: Upload → OCR → Field Detection → Selection → Blur → Export
"""

import requests
import json
import os
from pathlib import Path

def test_complete_ocr_flow_api():
    """Test the complete OCR flow through the web API."""
    print("🧪 Testing Complete OCR Flow API")
    print("=" * 50)
    
    try:
        # Use the sample document we created earlier
        sample_image_path = "sample_enhanced_document.png"
        
        if not os.path.exists(sample_image_path):
            print("❌ Sample document not found. Run test_web_api_enhanced.py first to create it.")
            return False
        
        print("1. 📄 Using existing sample document...")
        print(f"   ✅ Found: {sample_image_path}")
        
        # Test complete flow with auto-blur
        print("\\n2. 🚀 Testing complete flow with auto-blur...")
        
        with open(sample_image_path, 'rb') as f:
            files = {'file': ('test_document.png', f, 'image/png')}
            data = {
                'confidence_threshold': '0.3',
                'blur_strength': '20',
                'auto_blur': 'true'
            }
            
            response = requests.post(
                'http://localhost:5000/api/process-complete-flow',
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"   ✅ Complete flow successful!")
                print(f"   🆔 Flow ID: {result['flow_id']}")
                
                # Display processing summary
                summary = result['processing_summary']
                print(f"\\n📊 Processing Summary:")
                print(f"   ⏱️  Total Time: {summary['total_processing_time']:.2f}s")
                print(f"   📄 Text Length: {summary['text_length']} characters")
                print(f"   🔍 Fields Detected: {summary['fields_detected']}")
                print(f"   🎨 Fields Blurred: {summary['fields_blurred']}")
                print(f"   💫 Blur Strength: {summary['blur_strength_used']}")
                
                # Display step details
                step_details = result['step_details']
                print(f"\\n📋 Step Details:")
                print(f"   📤 Upload: {step_details['upload_validation']['file_size']} bytes")
                print(f"   👁️  OCR: {step_details['ocr_extraction']['word_count']} words")
                print(f"   🔍 Detection: {step_details['field_detection']['total_fields_found']} total, {step_details['field_detection']['fields_above_threshold']} above threshold")
                print(f"   ✅ Selection: {step_details['field_selection']['selection_method']} method")
                print(f"   🎨 Blur: {len(step_details['field_blurring']['blur_areas'])} areas blurred")
                
                # Save processed image
                processed_image_data = result.get('blurred_image_base64')
                if processed_image_data:
                    import base64
                    output_filename = f"complete_flow_output_{result['flow_id']}.png"
                    with open(output_filename, 'wb') as f:
                        f.write(base64.b64decode(processed_image_data))
                    print(f"\\n💾 Processed image saved: {output_filename}")
                
                # Show detected fields
                detected_fields = result.get('detected_fields', [])
                if detected_fields:
                    print(f"\\n🏷️  Detected Fields ({len(detected_fields)}):")
                    for i, field in enumerate(detected_fields[:5]):  # Show first 5
                        print(f"   {i+1}. {field['field_name']} - {field['field_value'][:30]}... (confidence: {field['confidence']:.2f})")
                    if len(detected_fields) > 5:
                        print(f"   ... and {len(detected_fields) - 5} more fields")
                
                # Show selected/blurred fields
                selected_fields = result.get('selected_fields', [])
                if selected_fields:
                    print(f"\\n🎯 Selected for Blurring ({len(selected_fields)}):")
                    for field in selected_fields:
                        print(f"   • {field['field_name']} (confidence: {field['confidence']:.2f})")
                
            else:
                print(f"   ❌ Flow failed: {result.get('error', 'Unknown error')}")
                if 'failed_step' in result:
                    print(f"   🚫 Failed at: {result['failed_step']}")
                return False
                
        else:
            print(f"   ❌ API call failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        # Test flow history
        print("\\n3. 📚 Testing flow history...")
        history_response = requests.get('http://localhost:5000/api/flow-history')
        
        if history_response.status_code == 200:
            history_result = history_response.json()
            if history_result.get('success'):
                flows = history_result.get('flow_history', [])
                print(f"   ✅ Retrieved {len(flows)} flow records")
                if flows:
                    latest_flow = flows[-1]
                    print(f"   📝 Latest flow: {latest_flow['flow_id']} at {latest_flow['timestamp']}")
            else:
                print(f"   ⚠️ History retrieval failed: {history_result.get('error')}")
        else:
            print(f"   ❌ History API failed: {history_response.status_code}")
        
        print("\\n🎉 COMPLETE OCR FLOW TEST SUCCESSFUL!")
        print("✅ Sequential processing working:")
        print("   1. Document Upload ✅")
        print("   2. OCR Text Extraction ✅") 
        print("   3. Field Detection ✅")
        print("   4. Field Selection ✅")
        print("   5. Field Blurring ✅")
        print("   6. Image Export ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_flow():
    """Test the complete flow using direct function call."""
    print("\\n🧪 Testing Direct Complete OCR Flow")
    print("=" * 50)
    
    try:
        from complete_ocr_flow import process_document_sequential
        
        sample_image_path = "sample_enhanced_document.png"
        
        if not os.path.exists(sample_image_path):
            print("❌ Sample document not found.")
            return False
        
        print("1. 🚀 Processing document through complete flow...")
        
        result = process_document_sequential(
            image_path=sample_image_path,
            selected_fields=None,  # Auto-select
            blur_strength=15,
            confidence_threshold=0.3
        )
        
        if result.get('success'):
            print(f"   ✅ Direct flow successful!")
            print(f"   🆔 Flow ID: {result['flow_id']}")
            
            summary = result['processing_summary']
            print(f"\\n📊 Summary: {summary['fields_detected']} detected, {summary['fields_blurred']} blurred in {summary['total_processing_time']:.2f}s")
            
            return True
        else:
            print(f"   ❌ Direct flow failed: {result.get('error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Direct flow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Complete OCR Flow Test Suite")
    print("=" * 60)
    
    # Test API
    api_success = test_complete_ocr_flow_api()
    
    # Test direct function
    direct_success = test_direct_flow()
    
    print("\\n" + "=" * 60)
    print("📋 Test Results:")
    print(f"   API Test: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"   Direct Test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    
    if api_success and direct_success:
        print("\\n🎉 ALL TESTS PASSED!")
        print("🚀 Complete OCR Flow is ready for production use!")
    else:
        print("\\n⚠️ Some tests failed. Please check the implementation.")
    
    exit(0 if (api_success and direct_success) else 1)