# AI-Powered Fake Job Verification System
# Complete implementation with ML models, web interface, and API
#=============================================================================
# 1. MAIN APPLICATION 
#=============================================================================

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
import re
import os
from datetime import datetime
import logging
import requests
from urllib.parse import urlparse
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data on startup
def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        import ssl
        
        # Handle SSL issues in some environments
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required data
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Download TextBlob corpora
        import textblob
        try:
            # This will download the required corpora
            from textblob import TextBlob
            TextBlob("test").sentences  # This will trigger download if needed
        except:
            pass
            
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")

# Download NLTK data at startup
download_nltk_data()

# Now import TextBlob after downloading data
try:
    from textblob import TextBlob
except ImportError:
    print("⚠️  TextBlob not available, using fallback analysis")
    TextBlob = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#=============================================================================
# 2. JOB ANALYZER CLASS
# This is like the AI's brain - it makes all the smart decisions
#=============================================================================

class JobAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models or create new ones"""
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('models/job_classifier.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Models loaded successfully")
        except FileNotFoundError:
            logger.info("Models not found, will train new ones")
            self.train_model()

    def preprocess_text(self, text):
        """Clean and preprocess job description text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def extract_features(self, job_data):
        """Extract features from job posting data"""
        features = {}
        
        # Text features
        description = job_data.get('description', '')
        title = job_data.get('title', '')
        company = job_data.get('company', '')
        
        # Combine text fields
        full_text = f"{title} {description} {company}"
        cleaned_text = self.preprocess_text(full_text)
        
        # Basic text statistics
        features['text_length'] = len(description)
        features['word_count'] = len(description.split()) if description else 0
        features['sentence_count'] = len(description.split('.')) if description else 0
        
        # Salary analysis
        salary = job_data.get('salary', '')
        features['has_salary'] = 1 if salary else 0
        features['salary_range'] = self.extract_salary_range(salary)
        
        # Company analysis
        features['company_length'] = len(company)
        features['has_company_email'] = 1 if self.validate_company_email(job_data.get('email', '')) else 0
        
        # Location analysis
        location = job_data.get('location', '')
        features['has_location'] = 1 if location else 0
        features['location_specificity'] = len(location.split(',')) if location else 0
        
        # Urgency indicators (red flags)
        urgency_keywords = ['urgent', 'immediate', 'asap', 'quick money', 'easy money']
        features['urgency_score'] = sum(1 for keyword in urgency_keywords if keyword in cleaned_text)
        
        # Quality indicators - with fallback if TextBlob is not available
        if TextBlob:
            try:
                grammar_blob = TextBlob(description)
                features['grammar_score'] = len(grammar_blob.sentences) / max(len(description.split()), 1)
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
                features['grammar_score'] = 0.5  # Default neutral score
        else:
            # Simple fallback grammar score
            features['grammar_score'] = 0.5 if description else 0
        
        return features, cleaned_text

    def extract_salary_range(self, salary_text):
        """Extract and validate salary information"""
        if not salary_text:
            return 0
        
        # Look for salary numbers
        numbers = re.findall(r'\d+', salary_text.replace(',', ''))
        if numbers:
            salary_num = int(numbers[0])
            # Reasonable salary range check
            if 20000 <= salary_num <= 500000:
                return 1  # Reasonable
            else:
                return -1  # Suspicious
        return 0

    def validate_company_email(self, email):
        """Validate company email domain"""
        if not email or '@' not in email:
            return False
        
        domain = email.split('@')[1].lower()
        
        # Common free email providers (red flag for companies)
        free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        
        return domain not in free_providers

    def analyze_job(self, job_data):
        """Main analysis function"""
        try:
            # Extract features
            features, cleaned_text = self.extract_features(job_data)
            
            # Prepare data for model prediction
            if self.vectorizer and self.model:
                # Vectorize text
                text_features = self.vectorizer.transform([cleaned_text])
                
                # Make prediction
                prediction = self.model.predict(text_features)[0]
                probability = self.model.predict_proba(text_features)[0]
                
                # Calculate risk score
                risk_score = self.calculate_risk_score(features, job_data)
                
                return {
                    'prediction': 'legitimate' if prediction == 1 else 'fake',
                    'confidence': float(max(probability)),
                    'risk_score': risk_score,
                    'risk_level': self.get_risk_level(risk_score),
                    'features': features,
                    'warnings': self.generate_warnings(features, job_data)
                }
            else:
                # Fallback analysis without ML model
                risk_score = self.calculate_risk_score(features, job_data)
                return {
                    'prediction': 'legitimate' if risk_score < 0.5 else 'fake',
                    'confidence': 0.7,
                    'risk_score': risk_score,
                    'risk_level': self.get_risk_level(risk_score),
                    'features': features,
                    'warnings': self.generate_warnings(features, job_data)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing job: {str(e)}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'risk_score': 1.0,
                'risk_level': 'high',
                'error': str(e)
            }

    def calculate_risk_score(self, features, job_data):
        """Calculate risk score based on various factors"""
        risk = 0.0
        
        # Text quality issues
        if features['text_length'] < 100:
            risk += 0.2
        if features['word_count'] < 20:
            risk += 0.15
        
        # Salary red flags
        if features['salary_range'] == -1:
            risk += 0.25
        
        # Company red flags
        if not features['has_company_email']:
            risk += 0.2
        if features['company_length'] < 3:
            risk += 0.15
        
        # Urgency red flags
        risk += features['urgency_score'] * 0.1
        
        # Grammar issues
        if features['grammar_score'] < 0.05:
            risk += 0.1
        
        return min(risk, 1.0)

    def get_risk_level(self, risk_score):
        """Convert risk score to risk level"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        else:
            return 'high'

    def generate_warnings(self, features, job_data):
        """Generate specific warnings based on analysis"""
        warnings = []
        
        if features['urgency_score'] > 0:
            warnings.append("Job posting contains urgency keywords that are common in scams")
        
        if not features['has_company_email']:
            warnings.append("Company uses free email provider instead of corporate domain")
        
        if features['text_length'] < 100:
            warnings.append("Job description is unusually short")
        
        if features['salary_range'] == -1:
            warnings.append("Salary amount seems unrealistic")
        
        if not features['has_location']:
            warnings.append("No specific location provided")
        
        return warnings

    def train_model(self):
        """Train the machine learning model with sample data"""
        logger.info("Training new model...")
        
        # Generate sample training data
        training_data = self.generate_training_data()
        
        # Prepare features
        X_text = []
        X_numerical = []
        y = []
        
        for data_point in training_data:
            features, cleaned_text = self.extract_features(data_point['job_data'])
            X_text.append(cleaned_text)
            X_numerical.append([
                features['text_length'],
                features['word_count'],
                features['has_salary'],
                features['salary_range'],
                features['has_company_email'],
                features['urgency_score'],
                features['grammar_score']
            ])
            y.append(data_point['label'])
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_text_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Train model (using text features only for simplicity)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_text_vectorized, y)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('models/job_classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info("Model training completed and saved")

    def generate_training_data(self):
        """Generate sample training data"""
        legitimate_jobs = [
            {
                'job_data': {
                    'title': 'Software Engineer',
                    'company': 'Tech Corp Inc',
                    'description': 'We are seeking a talented Software Engineer to join our development team. You will be responsible for designing, developing, and maintaining web applications using modern technologies. Requirements include 3+ years of experience with Python, JavaScript, and SQL.',
                    'salary': '$70,000 - $90,000',
                    'location': 'San Francisco, CA',
                    'email': 'hr@techcorp.com'
                },
                'label': 1
            },
            {
                'job_data': {
                    'title': 'Data Scientist',
                    'company': 'Analytics Solutions LLC',
                    'description': 'Join our data science team to help drive business decisions through data analysis and machine learning. We offer competitive salary, health benefits, and opportunities for professional growth. Requires PhD in Statistics or related field.',
                    'salary': '$95,000 - $120,000',
                    'location': 'New York, NY',
                    'email': 'careers@analyticsolutions.com'
                },
                'label': 1
            }
        ]
        
        fake_jobs = [
            {
                'job_data': {
                    'title': 'Easy Money From Home!!!',
                    'company': 'QuickCash',
                    'description': 'Make $5000 per week working from home! No experience required! Just send your personal information to get started immediately!',
                    'salary': '$5000 per week',
                    'location': '',
                    'email': 'money@gmail.com'
                },
                'label': 0
            },
            {
                'job_data': {
                    'title': 'Urgent Data Entry',
                    'company': 'FastEntry',
                    'description': 'URGENT! Need people for data entry. Pay $50/hour. Contact immediately!',
                    'salary': '$50/hour',
                    'location': 'Work from anywhere',
                    'email': 'jobs@yahoo.com'
                },
                'label': 0
            }
        ]
        
        return legitimate_jobs + fake_jobs

#=============================================================================
# 3. DATABASE SETUP
#=============================================================================

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('job_verification.db')
    cursor = conn.cursor()
    
    # Create jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            description TEXT NOT NULL,
            salary TEXT,
            location TEXT,
            email TEXT,
            prediction TEXT,
            confidence REAL,
            risk_score REAL,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create analysis_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            user_ip TEXT,
            analysis_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# This remembers everything the AI has analyzed
def save_job_analysis(job_data, analysis_result):
    """Save job analysis to database"""
    conn = sqlite3.connect('job_verification.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO jobs (title, company, description, salary, location, email,
                         prediction, confidence, risk_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        job_data.get('title', ''),
        job_data.get('company', ''),
        job_data.get('description', ''),
        job_data.get('salary', ''),
        job_data.get('location', ''),
        job_data.get('email', ''),
        analysis_result.get('prediction', ''),
        analysis_result.get('confidence', 0.0),
        analysis_result.get('risk_score', 0.0),
        analysis_result.get('risk_level', '')
    ))
    
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return job_id

#=============================================================================
# 4. FLASK ROUTES
#=============================================================================

# Initialize the job analyzer
analyzer = JobAnalyzer()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Job analysis page"""
    if request.method == 'POST':
        try:
            # Get form data
            job_data = {
                'title': request.form.get('title', ''),
                'company': request.form.get('company', ''),
                'description': request.form.get('description', ''),
                'salary': request.form.get('salary', ''),
                'location': request.form.get('location', ''),
                'email': request.form.get('email', '')
            }
            
            # Validate required fields
            if not all([job_data['title'], job_data['company'], job_data['description']]):
                flash('Please fill in at least the job title, company, and description fields.', 'error')
                return render_template('analyze.html', job_data=job_data)
            
            # Analyze the job
            result = analyzer.analyze_job(job_data)
            
            # Save to database
            job_id = save_job_analysis(job_data, result)
            
            return render_template('results.html',
                                 job_data=job_data,
                                 result=result,
                                 job_id=job_id)
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            flash(f'An error occurred during analysis: {str(e)}', 'error')
            return render_template('analyze.html')
    
    return render_template('analyze.html')

# This lets other websites talk to your AI
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for job analysis"""
    try:
        job_data = request.get_json()
        
        if not job_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['title', 'company', 'description']
        if not all(field in job_data for field in required_fields):
            return jsonify({'error': 'Missing required fields: title, company, description'}), 400
        
        # Analyze the job
        result = analyzer.analyze_job(job_data)
        
        # Save to database
        job_id = save_job_analysis(job_data, result)
        result['job_id'] = job_id
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        conn = sqlite3.connect('job_verification.db')
        cursor = conn.cursor()
        
        # Get total jobs analyzed
        cursor.execute('SELECT COUNT(*) FROM jobs')
        total_jobs = cursor.fetchone()[0]
        
        # Get fake job percentage
        cursor.execute('SELECT COUNT(*) FROM jobs WHERE prediction = "fake"')
        fake_jobs = cursor.fetchone()[0]
        
        # Get recent analyses
        cursor.execute('''
            SELECT title, company, prediction, risk_level, created_at
            FROM jobs
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        recent_jobs = cursor.fetchall()
        
        conn.close()
        
        stats = {
            'total_analyzed': total_jobs,
            'fake_percentage': (fake_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'legitimate_percentage': ((total_jobs - fake_jobs) / total_jobs * 100) if total_jobs > 0 else 0,
            'recent_analyses': [
                {
                    'title': job[0],
                    'company': job[1],
                    'prediction': job[2],
                    'risk_level': job[3],
                    'date': job[4]
                } for job in recent_jobs
            ]
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['GET', 'POST'])
def batch_analyze():
    """Batch analysis from CSV file"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and file.filename.endswith('.csv'):
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process CSV
                df = pd.read_csv(filepath)
                results = []
                
                for index, row in df.iterrows():
                    job_data = {
                        'title': row.get('title', ''),
                        'company': row.get('company', ''),
                        'description': row.get('description', ''),
                        'salary': row.get('salary', ''),
                        'location': row.get('location', ''),
                        'email': row.get('email', '')
                    }
                    
                    result = analyzer.analyze_job(job_data)
                    job_id = save_job_analysis(job_data, result)
                    
                    results.append({
                        'job_data': job_data,
                        'result': result,
                        'job_id': job_id
                    })
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('batch_results.html', results=results)
            
            else:
                flash('Please upload a CSV file', 'error')
                return redirect(request.url)
                
        except Exception as e:
            logger.error(f"Batch analysis error: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('batch_analyze.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

#=============================================================================
# 5. UTILITY FUNCTIONS
#=============================================================================

def create_sample_data():
    """Create sample CSV file for testing"""
    sample_data = [
        {
            'title': 'Software Engineer',
            'company': 'Google Inc',
            'description': 'We are looking for a talented software engineer to join our team. You will work on cutting-edge projects and collaborate with brilliant minds.',
            'salary': '$120,000 - $150,000',
            'location': 'Mountain View, CA',
            'email': 'careers@google.com'
        },
        {
            'title': 'Make Money Fast!!!',
            'company': 'EasyMoney',
            'description': 'Make $5000 per week from home! No experience needed! Contact us now!',
            'salary': '$5000/week',
            'location': 'Work from home',
            'email': 'money@gmail.com'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_jobs.csv', index=False)
    print("Sample data created: sample_jobs.csv")

#=============================================================================
# 6. MAIN EXECUTION
# This starts everything up, like turning on all the lights
#=============================================================================

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Create sample data file
    create_sample_data()
    
    # Get port from environment for Railway
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Run the app
    app.run(debug=debug_mode, host='0.0.0.0', port=port)