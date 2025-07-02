from flask import Flask
import os
import sys

# Add the parent directory to Python path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, init_database, create_sample_data

# Initialize for Vercel
try:
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    init_database()
    create_sample_data()
except:
    pass  # Skip errors on serverless

# Export for Vercel
def handler(request):
    return app(request.environ, lambda *args: None)

# Also support direct Flask app export
application = app

if __name__ == "__main__":
    app.run(debug=False)