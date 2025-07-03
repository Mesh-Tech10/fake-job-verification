#!/bin/bash
echo "🚀 Starting Railway build process..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads models logs nltk_data

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', download_dir='./nltk_data', quiet=True)
nltk.download('brown', download_dir='./nltk_data', quiet=True)
nltk.download('wordnet', download_dir='./nltk_data', quiet=True)
print('✅ NLTK data downloaded')
"

# Download TextBlob corpora
echo "📥 Setting up TextBlob..."
python -c "
try:
    import textblob
    from textblob import download_corpora
    download_corpora.download_all()
    print('✅ TextBlob corpora downloaded')
except Exception as e:
    print(f'⚠️  TextBlob setup warning: {e}')
"

echo "✅ Build completed successfully!"