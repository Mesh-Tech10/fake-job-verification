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
    print("\n" + "="*50)
    print("🌐 Application will be available at:")
    print("   http://localhost:5000")
    print("   http://127.0.0.1:5000")
    print("\n📋 Available endpoints:")
    print("   /          - Home page")
    print("   /analyze   - Single job analysis")
    print("   /batch_analyze - Batch CSV analysis")
    print("   /api/analyze - API endpoint")
    print("   /api/stats - Statistics API")
    print("="*50 + "\n")
    
    # Start the application
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()