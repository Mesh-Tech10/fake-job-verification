from flask import Flask, render_template, request, jsonify
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create simplified Flask app for demo
app = Flask(__name__, template_folder='../templates')
app.secret_key = 'demo-secret-key'

# Simple demo analyzer (without ML dependencies)
class DemoJobAnalyzer:
    def analyze_job(self, job_data):
        """Demo analysis without ML dependencies"""
        # Simple keyword-based analysis for demo
        description = job_data.get('description', '').lower()
        email = job_data.get('email', '').lower()
        salary = job_data.get('salary', '').lower()
        
        # Demo scoring logic
        score = 0.8  # Default good score
        warnings = []
        
        # Simple red flags
        if any(word in description for word in ['urgent', 'easy money', 'no experience']):
            score -= 0.3
            warnings.append("Contains urgency keywords")
        
        if any(domain in email for domain in ['gmail.com', 'yahoo.com', 'hotmail.com']):
            score -= 0.2
            warnings.append("Uses free email provider")
            
        if any(word in salary for word in ['$5000', 'guaranteed', 'per week']):
            score -= 0.4
            warnings.append("Unrealistic salary claims")
        
        # Determine prediction
        prediction = 'legitimate' if score > 0.5 else 'fake'
        risk_level = 'low' if score > 0.7 else 'medium' if score > 0.4 else 'high'
        
        return {
            'prediction': prediction,
            'confidence': min(score, 1.0),
            'risk_score': max(0, 1 - score),
            'risk_level': risk_level,
            'warnings': warnings,
            'demo_mode': True
        }

# Initialize demo analyzer
analyzer = DemoJobAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        job_data = {
            'title': request.form.get('title', ''),
            'company': request.form.get('company', ''),
            'description': request.form.get('description', ''),
            'salary': request.form.get('salary', ''),
            'location': request.form.get('location', ''),
            'email': request.form.get('email', '')
        }
        
        result = analyzer.analyze_job(job_data)
        return render_template('results.html', job_data=job_data, result=result)
    
    return render_template('analyze.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    job_data = request.get_json()
    result = analyzer.analyze_job(job_data)
    return jsonify(result)

@app.route('/batch_analyze')
def batch_analyze():
    return render_template('batch_analyze.html')

# Export for Vercel
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(debug=False)