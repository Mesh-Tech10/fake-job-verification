#!/usr/bin/env python3
"""
Script to find all Python imports in the project
"""

import os
import re

def find_imports_in_file(filepath):
    """Find all import statements in a Python file"""
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                # Match import statements
                if re.match(r'^(import|from)\s+', line):
                    imports.append((line_num, line))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return imports

def find_all_imports(directory='.'):
    """Find all imports in all Python files in the directory"""
    all_imports = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip virtual environment and other non-source directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                imports = find_imports_in_file(filepath)
                if imports:
                    all_imports[filepath] = imports
    
    return all_imports

def extract_package_names(imports_dict):
    """Extract unique package names from import statements"""
    packages = set()
    
    for filepath, imports in imports_dict.items():
        for line_num, import_line in imports:
            # Extract package name from import statement
            if import_line.startswith('from '):
                # from package import something
                package = import_line.split()[1].split('.')[0]
            elif import_line.startswith('import '):
                # import package
                package = import_line.split()[1].split('.')[0]
            else:
                continue
                
            # Filter out local imports and built-in modules
            if not package.startswith('.') and package not in ['os', 'sys', 'json', 'datetime', 'time', 're', 'collections', 'itertools', 'functools', 'math', 'random', 'urllib', 'logging', 'pathlib', 'typing', 'io']:
                packages.add(package)
    
    return sorted(packages)

if __name__ == '__main__':
    print("🔍 Finding all Python imports in the project...\n")
    
    # Find all imports
    imports_dict = find_all_imports()
    
    # Display all imports by file
    print("📁 All imports by file:")
    print("=" * 50)
    for filepath, imports in imports_dict.items():
        print(f"\n{filepath}:")
        for line_num, import_line in imports:
            print(f"  {line_num:3d}: {import_line}")
    
    # Extract and display unique packages
    packages = extract_package_names(imports_dict)
    
    print(f"\n\n📦 Unique third-party packages found:")
    print("=" * 50)
    for package in packages:
        print(f"  - {package}")
    
    print(f"\n🎯 Suggested requirements.txt additions:")
    print("=" * 50)
    
    # Common package mappings
    package_versions = {
        'flask': 'Flask==2.3.3',
        'sklearn': 'scikit-learn==1.3.0',
        'pandas': 'pandas==2.0.3',
        'numpy': 'numpy==1.24.3',
        'requests': 'requests==2.31.0',
        'nltk': 'nltk==3.8.1',
        'matplotlib': 'matplotlib==3.7.2',
        'seaborn': 'seaborn==0.12.2',
        'textblob': 'textblob==0.17.1',
        'joblib': 'joblib==1.3.2',
        'scipy': 'scipy==1.11.1',
        'plotly': 'plotly==5.15.0',
        'wordcloud': 'wordcloud==1.9.2',
        'gunicorn': 'gunicorn==21.2.0',
        'psycopg2': 'psycopg2-binary==2.9.7',
        'sqlalchemy': 'Flask-SQLAlchemy==3.0.5',
        'dotenv': 'python-dotenv==1.0.0',
    }
    
    for package in packages:
        if package.lower() in package_versions:
            print(package_versions[package.lower()])
        else:
            print(f"{package}  # Add version number")
    
    print(f"\n✅ Total packages found: {len(packages)}")