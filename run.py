#!/usr/bin/env python3
"""
Simple startup script for the AI Job Verification System
"""

import os
import sys

def main():
    """Main function to start the application"""
    print("🚀 Starting AI Job Verification System...")
    
    try:
        # Import from app.py file instead of app folder
        from app import app, init_database, create_sample_data
        print("✅ App imported successfully!")
        
        # Only initialize database and sample data in development
        if os.environ.get('RAILWAY_ENVIRONMENT') != 'production':
            # Initialize database
            print("📊 Initializing database...")
            init_database()
            
            # Create sample data
            print("📝 Creating sample data...")
            create_sample_data()
        
        # Check if models directory exists
        if not os.path.exists('models'):
            print("🤖 Creating models directory...")
            os.makedirs('models', exist_ok=True)
        
        print("✅ Setup complete!")
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        
        if not os.environ.get('RAILWAY_ENVIRONMENT'):
            print("\n" + "="*50)
            print("🌐 Application will be available at:")
            print(f"   http://localhost:{port}")
            print(f"   http://127.0.0.1:{port}")
            print("\n📋 Available endpoints:")
            print("   /          - Home page")
            print("   /analyze   - Single job analysis")
            print("   /batch_analyze - Batch CSV analysis")
            print("   /api/analyze - API endpoint")
            print("   /api/stats - Statistics API")
            print("="*50 + "\n")
        
        # Start the application
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Creating minimal Flask app as fallback...")
        
        # Fallback minimal app
        from flask import Flask
        fallback_app = Flask(__name__)
        
        @fallback_app.route('/')
        def hello():
            return "AI Job Verification System - Minimal Mode"
        
        port = int(os.environ.get('PORT', 5000))
        fallback_app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()