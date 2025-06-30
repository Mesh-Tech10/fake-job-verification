import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///job_verification.db'
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    
    # ML Model settings
    MODEL_PATH = 'models'
    VECTORIZER_FILE = 'vectorizer.pkl'
    CLASSIFIER_FILE = 'job_classifier.pkl'
    
    # API settings
    API_RATE_LIMIT = '100 per hour'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'app.log'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production
    
    # Production database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}