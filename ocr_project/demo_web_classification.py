#!/usr/bin/env python3
"""
Web Interface Demo for Document Classification
Demonstrates the complete document classification workflow through the web interface.
"""

import requests
import json
import time
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont
import io

class WebInterfaceDemo:
    """Demo class for testing the web interface."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def check_server_status(self):
        """Check if the Flask server is running."""
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def create_sample_document_image(self, doc_type: str, content: str):
        """Create a sample document image for testing."""
        # Create a simple document image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            title_font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw document header
        draw.text((50, 30), f"{doc_type.replace('_', ' ')}", fill='black', font=title_font)
        
        # Draw content
        lines = content.strip().split('\n')
        y_pos = 80
        for line in lines:
            if line.strip():
                draw.text((50, y_pos), line.strip(), fill='black', font=font)
                y_pos += 25
        
        # Save to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return img_buffer
    
    def test_document_upload(self, doc_type: str, content: str):
        """Test document upload and classification."""
        print(f"🧪 Testing {doc_type} Classification via Web Interface")
        print("-" * 55)
        
        # Create sample image
        img_buffer = self.create_sample_document_image(doc_type, content)
        
        # Prepare upload
        files = {
            'file': (f'test_{doc_type.lower()}.png', img_buffer, 'image/png')
        }
        
        try:
            # Upload and process
            start_time = time.time()
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=30)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    print(f"✅ Upload successful! Processing time: {processing_time:.3f}s")
                    
                    # Display document classifications
                    if 'document_classifications' in result:
                        print(f"📄 Document Classifications:")
                        for i, cls in enumerate(result['document_classifications'][:3], 1):
                            confidence = float(cls['confidence']) * 100
                            print(f"  {i}. {cls['document_type']}: {confidence:.1f}%")
                            if cls['keywords_found']:
                                print(f"     Keywords: {', '.join(cls['keywords_found'])}")
                    
                    # Display field suggestions
                    if 'rag_suggestions' in result:
                        print(f"🏷️ Field Suggestions ({len(result['rag_suggestions'])} found):")
                        for suggestion in result['rag_suggestions'][:5]:
                            confidence = float(suggestion['confidence']) * 100
                            print(f"  • {suggestion['field_name']} ({suggestion['field_category']}): {confidence:.1f}%")
                            if suggestion['suggested_value']:
                                print(f"    Value: {suggestion['suggested_value']}")
                    
                    # Display processing metadata
                    if 'processing_metadata' in result:
                        meta = result['processing_metadata']
                        if 'document_classification' in meta:
                            dc = meta['document_classification']
                            print(f"📊 Best Classification: {dc['best_type']} ({dc['confidence']})")
                    
                    return True
                else:
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Upload failed with status: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False
        finally:
            img_buffer.close()
    
    def run_comprehensive_demo(self):
        """Run comprehensive web interface demonstration."""
        print("🌐 WEB INTERFACE DOCUMENT CLASSIFICATION DEMO")
        print("=" * 60)
        
        # Check server status
        print("🔍 Checking server status...")
        if not self.check_server_status():
            print("❌ Flask server is not running!")
            print("   Please start the server with: python app.py")
            return False
        
        print("✅ Server is running!")
        print(f"🌐 Base URL: {self.base_url}")
        
        # Test documents
        test_documents = {
            "AADHAR_CARD": """
            GOVERNMENT OF INDIA
            UNIQUE IDENTIFICATION AUTHORITY OF INDIA
            AADHAAR
            
            Name: DEMO USER
            DOB: 15/08/1990
            Aadhaar Number: 1234 5678 9012
            Address: 123 Demo Street, Demo City 560001
            """,
            
            "PAN_CARD": """
            INCOME TAX DEPARTMENT
            GOVERNMENT OF INDIA
            PERMANENT ACCOUNT NUMBER CARD
            
            Name: DEMO TAXPAYER
            Father's Name: DEMO FATHER
            Date of Birth: 22/03/1985
            PAN: DEMOX1234Y
            """,
            
            "VOTER_ID": """
            ELECTION COMMISSION OF INDIA
            ELECTORAL PHOTO IDENTITY CARD
            
            Name: DEMO VOTER
            Father's Name: DEMO PARENT
            EPIC No.: DEM1234567
            Age: 35
            Address: Demo Constituency
            """,
            
            "PASSPORT": """
            PASSPORT
            REPUBLIC OF INDIA
            MINISTRY OF EXTERNAL AFFAIRS
            
            Name: DEMO TRAVELER
            Date of Birth: 10/06/1988
            Passport No: D1234567
            Place of Birth: DEMO CITY
            """
        }
        
        success_count = 0
        total_tests = len(test_documents)
        
        for doc_type, content in test_documents.items():
            success = self.test_document_upload(doc_type, content)
            if success:
                success_count += 1
            print()  # Empty line for spacing
        
        # Final results
        print("📈 DEMO RESULTS SUMMARY")
        print("=" * 35)
        print(f"✅ Successful Classifications: {success_count}/{total_tests}")
        print(f"📊 Success Rate: {(success_count/total_tests)*100:.1f}%")
        
        if success_count == total_tests:
            print("🎉 All document types classified successfully!")
            print("✨ Web interface is working perfectly!")
        else:
            print("⚠️ Some classifications failed. Check server logs.")
        
        return success_count == total_tests

def main():
    """Main demo function."""
    demo = WebInterfaceDemo()
    
    print("🚀 Starting Web Interface Demo...")
    print("📄 This demo tests document classification through the web interface")
    print()
    
    success = demo.run_comprehensive_demo()
    
    if success:
        print("\n🎯 DEMO COMPLETED SUCCESSFULLY!")
        print("💡 The document classification system is fully operational!")
        print("🌐 You can now upload documents through the web interface at:")
        print("   http://localhost:5000")
    else:
        print("\n❌ Demo completed with issues.")
        print("🔧 Please check the server logs and try again.")

if __name__ == "__main__":
    main()