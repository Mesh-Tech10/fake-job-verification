#!/usr/bin/env python3
"""
WSGI entry point for production deployment
"""

import os
from app import app

# Production configuration
if __name__ != '__main__':
    # This runs when deployed (not in debug mode)
    # Skip database initialization for production
    pass

# For gunicorn
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)