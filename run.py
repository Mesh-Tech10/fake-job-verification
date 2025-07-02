#!/usr/bin/env python3
"""
Simple startup script for the AI Job Verification System
"""

import os
import sys
from app import app, init_database, create_sample_data

def main():
    """Main function to start the application"""
    print("🚀 Starting AI Job Verification System...")
    
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
    try:
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()