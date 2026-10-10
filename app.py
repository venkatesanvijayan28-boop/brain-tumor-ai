"""
Brain Tumor Detection - Flask Web Application
Upload MRI images and get AI-powered tumor classification results.
"""

import os
import uuid
import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from predict import predict_tumor
from quality_checker import check_image_quality
from severity_analysis import analyze_severity
from voice_alert import speak_result
from report_generator import generate_report

# ============================================================
# App Configuration
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'brain-tumor-ai-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

# In-memory prediction history
prediction_history = []


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Home page with image upload form."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint to verify deployment status."""
    return jsonify({
        'status': 'healthy',
        'app': 'Brain Tumor AI'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please select an MRI image.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected. Please choose an MRI image.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400

    try:
        # Save uploaded file with unique name
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Step 1: Check image quality
        quality = check_image_quality(filepath)

        # Step 2: Make prediction (even if quality is poor, still predict)
        result = predict_tumor(filepath)

        # Step 3: Analyze severity
        severity = analyze_severity(result['label'], result['confidence'])

        # Step 4: Voice alert text generation
        use_voice = request.form.get('voice_alert', 'false').lower() == 'true'
        alert_text = f"Prediction result: {result['label']} with {result['confidence']:.1f}% confidence. Severity level: {severity.get('severity', 'Normal')}."
        if use_voice:
            try:
                speak_result(alert_text)
            except Exception:
                pass

        # Store in history
        history_entry = {
            'id': uuid.uuid4().hex[:8],
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': file.filename,
            'image_path': unique_filename,
            'label': result['label'],
            'confidence': result['confidence'],
            'severity': severity.get('severity', 'Unknown'),
            'risk_level': severity.get('risk_level', 'Unknown')
        }
        prediction_history.insert(0, history_entry)

        # Build response
        response = {
            'success': True,
            'prediction': {
                'label': result['label'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            },
            'severity': severity,
            'quality': quality,
            'image_url': f'/uploads/{unique_filename}',
            'history_id': history_entry['id'],
            'voice_text': alert_text if use_voice else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download a PDF report."""
    try:
        data = request.get_json()

        prediction_data = {
            'label': data.get('label', 'Unknown'),
            'confidence': data.get('confidence', 0),
            'severity': data.get('severity', {}),
            'quality': data.get('quality', {}),
            'image_path': data.get('image_path', '')
        }

        # Generate report
        report_filename = f"brain_tumor_report_{uuid.uuid4().hex[:8]}.pdf"
        report_path = os.path.join('static', 'reports', report_filename)
        generate_report(report_path, prediction_data)

        return send_file(report_path, as_attachment=True, download_name=report_filename)

    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


@app.route('/history')
def history():
    """View prediction history."""
    return render_template('history.html', history=prediction_history)


@app.route('/api/history')
def api_history():
    """Return prediction history as JSON."""
    return jsonify(prediction_history)


# ============================================================
# Run Application
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR DETECTION AI - WEB APPLICATION")
    print("=" * 60)
    print(f"  Server starting at http://127.0.0.1:5000")
    print(f"  Upload MRI images for tumor classification")
    print("=" * 60 + "\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
