#!/usr/bin/env python3
"""
Automatic fixer for app.py syntax errors.
This script will:
1. Remove duplicate code sections
2. Fix syntax errors
3. Backup original file
"""

import os
import shutil
from datetime import datetime

def fix_app_py():
    app_py_path = 'd:/Agile/ocr_project/app.py'
    
    # Backup original
    backup_path = f'd:/Agile/ocr_project/app.py.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f"📁 Creating backup: {backup_path}")
    shutil.copy(app_py_path, backup_path)
    
    # Read file
    with open(app_py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Original file: {len(lines)} lines")
    
    # Find duplicates and issues
    # The file has duplicate sections starting around line 1650
    # We need to keep only one copy and fix the syntax
    
    # Strategy: Find the last occurrence of "if __name__ == '__main__':" and keep everything before it once
    
    fixed_lines = []
    found_main = False
    last_main_index = -1
    
    # Find last occurrence of if __name__
    for i, line in enumerate(lines):
        if "if __name__ == '__main__':" in line:
            last_main_index = i
    
    print(f"🔍 Found 'if __name__' at line {last_main_index}")
    
    # Keep everything up to first duplicate section
    # The duplicate starts around line 1653 with function redefinitions
    
    #Find first occurrence of duplicate marker
    first_duplicate_start = -1
    for i in range(1650, min(1850, len(lines))):
        if i < len(lines) and '"""Analyze document structure based on detected fields."""' in lines[i]:
            if first_duplicate_start == -1:
                first_duplicate_start = i - 3  # Include the @app.route decorator
            else:
                # This is the duplicate
                print(f"⚠️ Found duplicate at line {i}")
                break
    
    if first_duplicate_start > 0:
        # Keep lines before first duplicate
        fixed_lines = lines[:first_duplicate_start]
        
        # Add lines from last occurrence forward
        if last_main_index > 0:
            # Find where to resume (after duplicates)
            resume_index = last_main_index - 20  # Back up a bit to find the right spot
            
            # Find the correct functions before if __name__
            for i in range(resume_index, last_main_index):
                if '@app.route' in lines[i] or 'def ' in lines[i]:
                    resume_index = i
                    break
            
            fixed_lines.extend(lines[resume_index:])
    else:
        print("❌ Could not find duplicate section automatically")
        print("Manual fixing required")
        return False
    
    print(f"✅ Fixed file: {len(fixed_lines)} lines")
    
    # Write fixed version
    with open(app_py_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ File has been fixed!")
    print(f"📋 Backup saved to: {backup_path}")
    
    return True

if __name__ == '__main__':
    print("🔧 Fixing app.py syntax errors...")
    print()
    
    try:
        success = fix_app_py()
        
        if success:
            print()
            print("=" * 60)
            print("✅ app.py has been fixed!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("1. Run: python -m py_compile app.py")
            print("2. If no errors, add the redaction API code from redaction_api_patch.py")
            print("3. Start the app: python app.py")
        else:
            print()
            print("=" * 60)
            print("⚠️ Automatic fix failed - manual intervention needed")
            print("=" * 60)
            print()
            print("Please manually:")
            print("1. Open app.py in your editor")
            print("2. Look for duplicate function definitions around line 1650-1850")
            print("3. Remove the duplicate section")
            print("4. Ensure there's only one 'if __name__ == \"__main__\":' block at the end")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Manual fixing required. See walkthrough.md for instructions.")
