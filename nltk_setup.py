#!/usr/bin/env python3
"""
NLTK setup script for Railway deployment
Run this before starting the main application
"""

import nltk
import ssl
import os
import sys

def setup_nltk():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        print("📥 Downloading NLTK data...")
        
        # Download required packages
        downloads = [
            'punkt',
            'brown', 
            'wordnet',
            'averaged_perceptron_tagger',
            'vader_lexicon'
        ]
        
        for package in downloads:
            try:
                nltk.download(package, quiet=True)
                print(f"✅ Downloaded {package}")
            except Exception as e:
                print(f"⚠️  Warning: Could not download {package}: {e}")
        
        # Setup TextBlob
        try:
            print("📥 Setting up TextBlob...")
            import textblob
            from textblob import TextBlob
            
            # This will download corpora if needed
            test_blob = TextBlob("This is a test.")
            _ = test_blob.sentences
            print("✅ TextBlob setup complete")
            
        except Exception as e:
            print(f"⚠️  TextBlob setup warning: {e}")
        
        print("✅ NLTK setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_nltk()
    if not success:
        sys.exit(1)