import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# Heavy ML imports are deferred to processing time to avoid startup crashes
# when mediapipe/whisper aren't fully available on the deploy platform yet.

app = Flask(__name__)

# CORS: Allow frontend origins from environment variable, or allow all in development
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app, origins=ALLOWED_ORIGINS.split(','))

# Configuration
UPLOAD_FOLDER = Path('uploaded_audio')
OUTPUT_FOLDER = Path('.')
ALLOWED_EXTENSIONS = {'mp3', 'mpeg'}

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Store job status
jobs = {}

# Store Whisper model globally to avoid reloading
whisper_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_audio(job_id, audio_path, avatar_path, output_path, keypoints_dir):
    """Process audio file in background thread with intermediate output capture"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 5
        jobs[job_id]['message'] = 'Starting pipeline...'
        jobs[job_id]['intermediate_outputs'] = {
            'transcribed_text': None,
            'gloss': None,
            'video_sequence': None,
            'keypoint_files': None
        }

        # Lazy imports — heavy ML modules loaded only when processing starts
        import whisper
        from gloss_converter import to_gloss
        from gloss_to_video import gloss_to_video_sequence
        from keypoint_extractor import extract_keypoints
        from keypoint_retarget import retarget
        
        # 1) AUDIO -> TEXT
        jobs[job_id]['progress'] = 10
        jobs[job_id]['message'] = 'Transcribing audio to text...'
        whisper_model_name = os.environ.get('WHISPER_MODEL', 'medium')
        print(f"[INFO] Loading Whisper model: {whisper_model_name}")
        model = whisper.load_model(whisper_model_name)
        
        print(f"[INFO] Transcribing audio: {audio_path}")
        result = model.transcribe(str(audio_path), task="transcribe", language="en", verbose=False)
        text = (result.get("text") or "").strip()
        if not text:
            raise RuntimeError("Whisper transcription returned empty text.")
        
        jobs[job_id]['intermediate_outputs']['transcribed_text'] = text
        jobs[job_id]['progress'] = 25
        jobs[job_id]['message'] = 'Text transcription complete'
        print("[INFO] Final text output:", text)
        
        # 2) TEXT -> GLOSS
        jobs[job_id]['progress'] = 30
        jobs[job_id]['message'] = 'Converting text to gloss...'
        gloss = to_gloss(text)
        if not gloss:
            raise RuntimeError("Gloss converter returned empty gloss string.")
        
        jobs[job_id]['intermediate_outputs']['gloss'] = gloss
        jobs[job_id]['progress'] = 45
        jobs[job_id]['message'] = 'Gloss conversion complete'
        print("[INFO] Final gloss output:", gloss)
        
        # 3) GLOSS -> VIDEO SEQUENCE
        jobs[job_id]['progress'] = 50
        jobs[job_id]['message'] = 'Mapping gloss to video sequence...'
        video_list = gloss_to_video_sequence(gloss)
        print("[INFO] Video sequence:", video_list)
        
        if not video_list:
            raise RuntimeError("No videos found for the produced gloss; cannot continue.")
        
        # Store video filenames for display
        video_filenames = [Path(v).name for v in video_list]
        jobs[job_id]['intermediate_outputs']['video_sequence'] = video_filenames
        jobs[job_id]['progress'] = 60
        jobs[job_id]['message'] = 'Video sequence mapped'
        
        # 4) VIDEOS -> KEYPOINTS (Mediapipe)
        keypoints_dir.mkdir(parents=True, exist_ok=True)
        jobs[job_id]['progress'] = 65
        jobs[job_id]['message'] = 'Extracting keypoints from videos...'
        print(f"[INFO] Extracting keypoints into: {keypoints_dir}")
        keypoint_files = extract_keypoints(
            video_paths=[str(v) for v in video_list],
            output_dir=str(keypoints_dir),
            fps=30,
        )
        
        if not keypoint_files:
            raise RuntimeError("Keypoint extraction produced no JSON files.")
        
        # Store keypoint filenames for display
        keypoint_filenames = [Path(kf).name for kf in keypoint_files]
        jobs[job_id]['intermediate_outputs']['keypoint_files'] = keypoint_filenames
        jobs[job_id]['progress'] = 80
        jobs[job_id]['message'] = 'Keypoint extraction complete'
        print("[INFO] Keypoint data saved at:", keypoint_files)
        
        # Select the keypoints file to use (last one in the list)
        if not keypoint_files:
            raise RuntimeError("No keypoints JSON files were produced; cannot retarget avatar.")
        keypoints_path = Path(keypoint_files[-1])
        
        # 5) KEYPOINTS -> AVATAR GLTF ANIMATION
        jobs[job_id]['progress'] = 85
        jobs[job_id]['message'] = 'Retargeting keypoints to avatar...'
        print(f"[INFO] Retargeting keypoints from {keypoints_path} onto avatar {avatar_path}")
        retarget(
            gltf_path=avatar_path,
            keypoints_path=keypoints_path,
            mapping_path=None,
            output_path=output_path,
            animation_name='AutoGesture',
            fps=30.0,
        )
        
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Processing complete!'
        print(f"[SUCCESS] Animation 'AutoGesture' embedded into {output_path}")
        
        # Mark as completed
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['output_file'] = str(output_path)
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Error: {str(e)}'
        print(f"Error processing audio: {traceback.format_exc()}")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only MP3 files are allowed.'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    audio_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
    file.save(audio_path)
    
    # Set up paths
    avatar_path = Path('avatar.gltf')
    output_filename = f"avatar_with_anim_{job_id}.gltf"
    output_path = OUTPUT_FOLDER / output_filename
    keypoints_dir = Path('keypoints_out') / job_id
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize job status
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'File uploaded, queued for processing...',
        'output_file': None,
        'error': None,
        'intermediate_outputs': {
            'transcribed_text': None,
            'gloss': None,
            'video_sequence': None,
            'keypoint_files': None
        }
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_audio,
        args=(job_id, audio_path, avatar_path, output_path, keypoints_dir)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'File uploaded successfully'
    }), 200


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get processing status for a job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'intermediate_outputs': job.get('intermediate_outputs', {})
    }
    
    if job['status'] == 'completed':
        response['output_file'] = job['output_file']
    elif job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    
    return jsonify(response), 200


@app.route('/download/<job_id>', methods=['GET'])
def download_file(job_id):
    """Download the generated GLTF file"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'File not ready yet'}), 400
    
    output_file = Path(job['output_file'])
    if not output_file.exists():
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f'avatar_with_anim_{job_id}.gltf',
        mimetype='model/gltf+json'
    )


@app.route('/gltf/<job_id>', methods=['GET'])
def serve_gltf(job_id):
    """Serve the generated GLTF file for direct loading"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'File not ready yet'}), 400
    
    output_file = Path(job['output_file'])
    if not output_file.exists():
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_file,
        mimetype='model/gltf+json'
    )


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    # Check if avatar.gltf exists
    if not Path('avatar.gltf').exists():
        print("[WARNING] avatar.gltf not found. Make sure it exists in the current directory.")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
