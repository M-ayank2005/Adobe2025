#!/usr/bin/env python3
"""
Quick validation script to verify JSON output format
"""

import json
import sys
from pathlib import Path

def validate_json_format(json_file):
    """Validate that JSON file matches required format"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required fields
        if 'title' not in data:
            print(f"❌ Missing 'title' field in {json_file}")
            return False
        
        if 'outline' not in data:
            print(f"❌ Missing 'outline' field in {json_file}")
            return False
        
        # Check outline structure
        for i, item in enumerate(data['outline']):
            if 'level' not in item:
                print(f"❌ Missing 'level' field in outline item {i} in {json_file}")
                return False
            
            if 'text' not in item:
                print(f"❌ Missing 'text' field in outline item {i} in {json_file}")
                return False
            
            if 'page' not in item:
                print(f"❌ Missing 'page' field in outline item {i} in {json_file}")
                return False
            
            # Validate level format
            if item['level'] not in ['H1', 'H2', 'H3']:
                print(f"❌ Invalid level '{item['level']}' in outline item {i} in {json_file}")
                return False
            
            # Validate page number
            if not isinstance(item['page'], int) or item['page'] < 1:
                print(f"❌ Invalid page number '{item['page']}' in outline item {i} in {json_file}")
                return False
        
        print(f"✅ {json_file}: Valid format")
        print(f"   Title: {data['title']}")
        print(f"   Outline items: {len(data['outline'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {json_file}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating {json_file}: {e}")
        return False

def main():
    """Validate all JSON files in the output directory"""
    output_dir = Path("../Challenge_1a/sample_dataset/outputs")
    
    if not output_dir.exists():
        print(f"❌ Output directory {output_dir} does not exist")
        sys.exit(1)
    
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {output_dir}")
        sys.exit(1)
    
    print(f"🔍 Validating {len(json_files)} JSON files...\n")
    
    all_valid = True
    for json_file in sorted(json_files):
        if not validate_json_format(json_file):
            all_valid = False
        print()
    
    if all_valid:
        print("🎉 All JSON files are valid!")
        sys.exit(0)
    else:
        print("❌ Some JSON files have validation errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
