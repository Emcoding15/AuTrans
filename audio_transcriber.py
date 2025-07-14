import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QLabel,
                           QProgressBar, QFileDialog, QGroupBox, QSpinBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import whisper
from datetime import timedelta, datetime
from pyannote.audio import Pipeline
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import time

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

class TranscriptionThread(QThread):
    finished = pyqtSignal(tuple)  # Will emit (original_text, translated_text, minutes)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    chunk_progress = pyqtSignal(str)  # New signal for chunk processing updates
    
    def __init__(self, audio_file, num_speakers=2, chunk_size_seconds=600):
        super().__init__()
        self.audio_file = audio_file
        self.num_speakers = num_speakers
        self.speaker_encoder = None  # Lazy loading
        self.ollama_url = "http://localhost:11434/api/generate"
        self.executor = ThreadPoolExecutor(max_workers=3)  # Parallel processing
        self.cache = {}  # Simple cache for repeated operations
        self._stop_requested = False
        self.chunk_size_seconds = chunk_size_seconds  # Default 10-minute chunks
        self.checkpoint_dir = "transcription_checkpoints"
        
        # Initialize NLTK components once
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Cache stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def stop(self):
        """Request the thread to stop"""
        self._stop_requested = True
        
    def format_timestamp(self, seconds):
        return str(timedelta(seconds=round(seconds)))
        
    def assign_speakers(self, transcript_segments, num_speakers):
        """Fallback method for speaker assignment when diarization fails"""
        result = []
        current_speaker = 0
        min_switch_duration = 2.0  # Minimum duration before switching speakers
        last_switch_time = 0
        
        for segment in transcript_segments:
            current_time = segment['start']
            duration = segment['end'] - segment['start']
            
            # Switch speaker if enough time has passed
            if current_time - last_switch_time > min_switch_duration:
                current_speaker = (current_speaker + 1) % num_speakers
                last_switch_time = current_time
            
            result.append({
                'speaker': current_speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'] + " [auto-assigned]"  # Mark as automatically assigned
            })
            
        return result
        
    def get_speaker_diarization(self, audio_file):
        """Get speaker diarization with improved error handling and retries"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Initialize pipeline with error handling
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization@2.1",
                        use_auth_token=True
                    )
                except Exception as e:
                    if "authentication" in str(e).lower():
                        self.error.emit("Authentication error: Please check your HuggingFace token")
                        return None
                    raise
                    
                # Configure pipeline parameters
                diarization = pipeline(
                    audio_file,
                    num_speakers=self.num_speakers,
                    min_speakers=self.num_speakers,
                    max_speakers=self.num_speakers
                )
                
                # Validate diarization output
                if diarization is None or not hasattr(diarization, 'itertracks'):
                    raise ValueError("Invalid diarization output")
                    
                return diarization
                
            except Exception as e:
                error_msg = str(e)
                if "CUDA" in error_msg:
                    self.error.emit("GPU error detected. Falling back to CPU...")
                    torch.cuda.empty_cache()
                    continue
                    
                if attempt < max_retries - 1:
                    print(f"Diarization attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(retry_delay)
                else:
                    print(f"All diarization attempts failed: {error_msg}")
                    self.error.emit(f"Speaker diarization failed after {max_retries} attempts. Using fallback method.")
                    return None
                    
        return None
            
    def create_default_segments(self, audio_file):
        """Create default speaker segments when diarization fails"""
        try:
            # Get audio duration using torchaudio
            waveform, sample_rate = torchaudio.load(audio_file)
            duration = waveform.shape[1] / sample_rate
            
            # Create segments alternating between speakers every 3 seconds
            segments = []
            segment_duration = 3.0
            current_time = 0.0
            current_speaker = 0
            
            while current_time < duration:
                end_time = min(current_time + segment_duration, duration)
                segments.append({
                    'start': float(current_time),
                    'end': float(end_time),
                    'speaker': current_speaker
                })
                current_time = end_time
                current_speaker = (current_speaker + 1) % self.num_speakers
                
            return segments
            
        except Exception as e:
            print(f"Error creating default segments: {str(e)}")
            return None

    def force_speaker_separation(self, speaker_segments):
        """Force separation of speakers based on timing and duration"""
        if not speaker_segments:
            return None
            
        result = []
        current_speaker = 0
        min_switch_duration = 2.0  # Minimum duration before considering speaker change
        last_switch_time = speaker_segments[0]['start']
        
        # Sort segments by start time
        speaker_segments.sort(key=lambda x: x['start'])
        
        for segment in speaker_segments:
            duration = segment['end'] - segment['start']
            gap_since_last = segment['start'] - last_switch_time
            
            # Switch speaker if enough time has passed or segment is long enough
            if gap_since_last > min_switch_duration or duration > 3.0:
                current_speaker = (current_speaker + 1) % self.num_speakers
                last_switch_time = segment['start']
            
            result.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': current_speaker
            })
            
        return result

    def get_speaker_embeddings(self, audio_file, segment):
        """Get speaker embeddings with lazy loading and error handling"""
        try:
            if self.speaker_encoder is None:
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"}
                )
            
            # Load audio segment
            waveform, sample_rate = torchaudio.load(audio_file)
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            # Validate segment boundaries
            if start_sample >= waveform.shape[1] or end_sample <= start_sample:
                return None
                
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Get speaker embedding
            with torch.no_grad():  # Reduce memory usage
                embedding = self.speaker_encoder.encode_batch(segment_waveform)
                return embedding.squeeze().cpu().numpy()
                
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
            
    def compare_speakers(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return 0.0
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
        
    def assign_speakers_to_transcript(self, transcript_segments, speaker_segments):
        """Assign speakers to transcript segments using diarization or fallback method"""
        try:
            # If no speaker segments, use fallback method
            if not speaker_segments:
                return self.assign_speakers(transcript_segments, self.num_speakers)
                
            result = []
            for segment in transcript_segments:
                if self._stop_requested:
                    return []
                    
                # Get segment times
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                # Skip invalid segments
                if not text or end <= start:
                    continue
                
                # Find the most likely speaker for this segment
                speaker = self.get_speaker_for_segment(start, end, speaker_segments)
                
                result.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'text': text
                })
            
            return result
            
        except Exception as e:
            print(f"Error assigning speakers: {str(e)}")
            # Fallback to basic speaker assignment
            return self.assign_speakers(transcript_segments, self.num_speakers)

    def format_transcript(self, speaker_segments):
        """Format transcript with speaker labels and timestamps"""
        try:
            if not speaker_segments:
                return "No transcription available."
                
            formatted_text = ""
            current_speaker = None
            
            for segment in speaker_segments:
                timestamp = f"[{self.format_timestamp(segment['start'])} -> {self.format_timestamp(segment['end'])}]"
                speaker = segment.get('speaker', 0)
                text = segment.get('text', '').strip()
                
                # Add a blank line between different speakers
                if speaker != current_speaker:
                    if formatted_text:  # Don't add blank line at the start
                        formatted_text += "\n"
                    formatted_text += f"Speaker {speaker + 1}:\n"
                    current_speaker = speaker
                
                formatted_text += f"{timestamp} {text}\n"
            
            return formatted_text
            
        except Exception as e:
            print(f"Error formatting transcript: {str(e)}")
            return "Error formatting transcript."

    def extract_action_items(self, text):
        """Extract potential action items from text"""
        action_patterns = [
            r"need[s]? to\s+([^.!?]*)[.!?]",
            r"should\s+([^.!?]*)[.!?]",
            r"will\s+([^.!?]*)[.!?]",
            r"must\s+([^.!?]*)[.!?]",
            r"going to\s+([^.!?]*)[.!?]",
            r"task[s]?:?\s+([^.!?]*)[.!?]"
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action_items.append(match.group(0))
        return action_items

    def extract_key_points(self, text):
        """Extract key points from text"""
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        # Score sentences based on word importance
        word_freq = Counter()
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w not in stop_words and w.isalnum()]
            word_freq.update(words)
        
        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = sum(word_freq[w] for w in words if w not in stop_words and w.isalnum())
            sentence_scores.append((score, sentence))
        
        # Get top 5 sentences as key points
        key_points = [s[1] for s in sorted(sentence_scores, reverse=True)[:5]]
        return key_points

    @lru_cache(maxsize=128)  # Cache for repeated text analysis
    def analyze_text_segment(self, text):
        """Analyze text segment with caching"""
        words = word_tokenize(text.lower())
        return [w for w in words if w not in self.stop_words and w.isalnum()]
    
    def generate_meeting_minutes(self, transcript_segments):
        """Generate minutes with better timeout handling"""
        try:
            if not transcript_segments:
                return "No transcript content available to generate minutes."
            
            # Test Ollama connection first
            try:
                test_response = requests.get(self.ollama_url.replace("/generate", "/version"))
                print(f"Ollama server status: {test_response.status_code}")
                print(f"Ollama version: {test_response.json() if test_response.status_code == 200 else 'Not available'}")
            except Exception as e:
                print(f"Failed to connect to Ollama server: {str(e)}")
            
            # Rest of the existing code...
            full_text = " ".join([segment.get('text', '') for segment in transcript_segments])
            
            prompt = f"""
            Create detailed meeting minutes from this transcript.
            Focus on key points, decisions, and action items.
            Be concise but comprehensive.

            Transcript:
            {full_text}

            Format the output with:
            1. Executive Summary (2-3 sentences)
            2. Key Discussion Points 
            3. Decisions Made 
            4. Action Items 
            5. Next Steps 
            """
            
            print("Attempting to generate minutes...")
            for attempt in range(3):
                try:
                    print(f"\nAttempt {attempt + 1}:")
                    print(f"Sending request to {self.ollama_url}")
                    
                    response = requests.post(
                        self.ollama_url,
                        json={
                            "model": "llama2",
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                                "num_ctx": 4096
                            }
                        },
                        timeout=120
                    )
                    
                    print(f"Response status code: {response.status_code}")
                    if response.status_code == 200:
                        print("Successfully generated minutes")
                        return self.format_minutes(response.json()['response'])
                    
                    print(f"Failed attempt {attempt + 1}. Response: {response.text[:200]}...")
                    time.sleep(5)
                    
                except requests.Timeout:
                    print(f"Timeout on attempt {attempt + 1}")
                    continue
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {str(e)}")
                    print(f"Error type: {type(e)}")
                    if attempt < 2:
                        time.sleep(5)
                    continue
            
            return "Failed to generate meeting minutes after multiple attempts.\nDebug info:\n" + \
                   "1. Check if Ollama is running (ollama serve)\n" + \
                   "2. Check if llama2 model is installed (ollama list)\n" + \
                   "3. Try running: curl http://localhost:11434/api/generate -d '{\"model\": \"llama2\", \"prompt\": \"hello\"}'"
            
        except Exception as e:
            print(f"Critical error in minutes generation: {e}")
            return f"Error generating minutes: {str(e)}"

    def format_minutes(self, ai_response):
        """Format the final minutes with metadata"""
        try:
            minutes = "MEETING MINUTES\n"
            minutes += "=" * 50 + "\n\n"
            
            # Add metadata
            minutes += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            minutes += f"Number of Participants: {self.num_speakers}\n"
            minutes += "=" * 50 + "\n\n"
            
            # Add AI-generated content
            minutes += ai_response.strip() + "\n\n"
            
            return minutes
            
        except Exception as e:
            print(f"Error formatting minutes: {e}")
            return ai_response  # Return raw response if formatting fails

    def process_transcription(self, result, diarization):
        """Process transcription results with improved error handling"""
        try:
            if not result or 'segments' not in result:
                raise ValueError("Invalid transcription result")
                
            segments = []
            
            for segment in result['segments']:
                if self._stop_requested:
                    return []
                    
                text = segment.get('text', '').strip()
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                
                # Skip invalid segments
                if not text or end <= start:
                    continue
                    
                # Find the most likely speaker
                speaker = self.get_speaker_for_segment(start, end, diarization)
                
                segments.append({
                    'text': text,
                    'start': start,
                    'end': end,
                    'speaker': speaker if speaker is not None else 0
                })
            
            return segments
            
        except Exception as e:
            self.error.emit(f"Error processing transcription: {str(e)}")
            return []

    def get_speaker_for_segment(self, start, end, speaker_segments):
        """Determine the speaker for a given segment using speaker diarization data"""
        try:
            # Get the midpoint of the segment
            mid_time = (start + end) / 2
            
            # If we have speaker segments from diarization
            if hasattr(speaker_segments, 'itertracks'):
                # Use pyannote.audio diarization format
                for turn, _, speaker in speaker_segments.itertracks(yield_label=True):
                    if turn.start <= mid_time <= turn.end:
                        # Convert speaker label to integer index
                        try:
                            speaker_idx = int(speaker.split("_")[-1]) - 1
                            return min(speaker_idx, self.num_speakers - 1)
                        except (ValueError, IndexError):
                            return 0
            else:
                # Use basic speaker segments format
                for segment in speaker_segments:
                    if segment.get('start', 0) <= mid_time <= segment.get('end', float('inf')):
                        speaker = segment.get('speaker', 0)
                        return min(speaker, self.num_speakers - 1)
            
            # If no matching segment found, return default speaker
            return 0
            
        except Exception as e:
            print(f"Error getting speaker for segment: {e}")
            return 0

    def get_audio_duration(self, audio_file):
        """Get the duration of an audio file in seconds"""
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            duration = waveform.shape[1] / sample_rate
            return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0
            
    def split_audio(self, audio_file, output_dir):
        """Split a long audio file into chunks"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Get audio duration
            duration = self.get_audio_duration(audio_file)
            if duration <= 0:
                raise ValueError("Could not determine audio duration")
                
            # Calculate number of chunks
            num_chunks = int(np.ceil(duration / self.chunk_size_seconds))
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Create chunks
            chunk_files = []
            for i in range(num_chunks):
                if self._stop_requested:
                    break
                    
                # Calculate chunk boundaries
                start_time = i * self.chunk_size_seconds
                end_time = min((i + 1) * self.chunk_size_seconds, duration)
                
                # Convert time to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Extract chunk
                chunk = waveform[:, start_sample:end_sample]
                
                # Create 5-second overlap with previous chunk if not the first chunk
                if i > 0:
                    overlap_seconds = 5
                    overlap_samples = int(overlap_seconds * sample_rate)
                    start_sample = max(0, start_sample - overlap_samples)
                    overlap_chunk = waveform[:, start_sample:end_sample]
                    chunk = overlap_chunk
                
                # Save chunk
                chunk_file = os.path.join(output_dir, f"chunk_{i:03d}.wav")
                torchaudio.save(chunk_file, chunk, sample_rate)
                chunk_files.append((chunk_file, start_time, end_time))
                
                # Update progress
                progress_pct = int((i + 1) / num_chunks * 20)  # 20% of total progress for splitting
                self.progress.emit(progress_pct)
                self.chunk_progress.emit(f"Splitting audio: chunk {i+1}/{num_chunks}")
                
            return chunk_files
            
        except Exception as e:
            print(f"Error splitting audio: {e}")
            self.error.emit(f"Error splitting audio: {str(e)}")
            return []
            
    def create_checkpoint(self, chunk_index, results):
        """Save processing checkpoint"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                
            checkpoint_file = os.path.join(
                self.checkpoint_dir, 
                f"{os.path.basename(self.audio_file)}_checkpoint_{chunk_index}.json"
            )
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            return checkpoint_file
            
        except Exception as e:
            print(f"Error creating checkpoint: {e}")
            return None
            
    def load_checkpoint(self, chunk_index):
        """Load processing checkpoint if available"""
        try:
            checkpoint_file = os.path.join(
                self.checkpoint_dir, 
                f"{os.path.basename(self.audio_file)}_checkpoint_{chunk_index}.json"
            )
            
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def merge_transcriptions(self, chunk_results):
        """Merge transcription results from multiple chunks"""
        try:
            merged_original = []
            merged_translation = []
            merged_segments = []
            
            # Sort chunks by start time
            sorted_results = sorted(chunk_results, key=lambda x: x.get('start_time', 0))
            
            for chunk_result in sorted_results:
                # Process original text segments
                original_segments = chunk_result.get('original_segments', [])
                for segment in original_segments:
                    # Adjust timestamps by chunk start time
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] += chunk_result.get('start_time', 0)
                    adjusted_segment['end'] += chunk_result.get('start_time', 0)
                    merged_segments.append(adjusted_segment)
                    
                # Also store raw text results
                if 'original_text' in chunk_result:
                    # Add chunk info header
                    chunk_header = f"\n--- Chunk starting at {self.format_timestamp(chunk_result.get('start_time', 0))} ---\n"
                    merged_original.append(chunk_header + chunk_result['original_text'])
                    
                if 'translated_text' in chunk_result:
                    # Add chunk info header
                    chunk_header = f"\n--- Chunk starting at {self.format_timestamp(chunk_result.get('start_time', 0))} ---\n"
                    merged_translation.append(chunk_header + chunk_result['translated_text'])
            
            # Format final merged outputs
            final_original = "\n".join(merged_original)
            final_translation = "\n".join(merged_translation)
            
            # Generate minutes from all segments
            minutes = self.generate_meeting_minutes(merged_segments)
            
            return (final_original, final_translation, minutes, merged_segments)
            
        except Exception as e:
            print(f"Error merging transcriptions: {e}")
            self.error.emit(f"Error merging transcriptions: {str(e)}")
            return ("Error merging transcriptions", "Error merging transcriptions", "Error generating minutes", [])
            
    def process_audio_chunk(self, chunk_info, model):
        """Process a single audio chunk with full pipeline"""
        try:
            chunk_file, start_time, end_time = chunk_info
            chunk_index = int(os.path.basename(chunk_file).split('_')[1].split('.')[0])
            
            # Check if checkpoint exists
            checkpoint = self.load_checkpoint(chunk_index)
            if checkpoint is not None:
                self.chunk_progress.emit(f"Loaded checkpoint for chunk {chunk_index}")
                return checkpoint
            
            # Get speaker diarization for this chunk
            self.chunk_progress.emit(f"Diarizing speakers for chunk {chunk_index}")
            speaker_segments = self.get_speaker_diarization(chunk_file)
            
            if speaker_segments is None:
                self.chunk_progress.emit(f"Using fallback speaker assignment for chunk {chunk_index}")
                speaker_segments = []  # Will trigger fallback in assign_speakers_to_transcript
            
            # First detect the language
            self.chunk_progress.emit(f"Detecting language for chunk {chunk_index}")
            try:
                initial_result = model.transcribe(
                    chunk_file,
                    language=None,
                    word_timestamps=True,
                    task="transcribe"
                )
                
                # Ensure initial_result is a dictionary
                if not isinstance(initial_result, dict):
                    raise ValueError("Transcription result is not in the expected format")
                    
                detected_language = initial_result.get("language", "en")
                if not detected_language:
                    detected_language = "en"
                    
                print(f"Detected language for chunk {chunk_index}: {detected_language}")
            except Exception as e:
                print(f"Language detection failed for chunk {chunk_index}: {str(e)}, defaulting to English")
                detected_language = "en"
            
            # First transcription in original language (English/Tagalog)
            self.chunk_progress.emit(f"Transcribing chunk {chunk_index} in original language")
            # If detected language is not English or Tagalog, default to English
            if detected_language not in ["en", "tl"]:
                detected_language = "en"
                
            try:
                result_original = model.transcribe(
                    chunk_file,
                    task="transcribe",
                    language=detected_language,
                    word_timestamps=True,
                    initial_prompt="This audio may contain English and/or Tagalog language with multiple speakers."
                )
                
                if not isinstance(result_original, dict) or "segments" not in result_original:
                    raise ValueError("Invalid transcription result format")
                
                # Process segments with speaker diarization
                speaker_segments_with_text = self.assign_speakers_to_transcript(
                    result_original.get("segments", []),  # Use get() with default empty list
                    speaker_segments
                )
                
                if not speaker_segments_with_text:
                    raise ValueError("No valid segments found in transcription")
                    
                original_text = self.format_transcript(speaker_segments_with_text)
                
                # Add language detection info to original text
                lang_info = "Detected Language: " + ("English" if detected_language == "en" else "Tagalog/English" if detected_language == "tl" else "Mixed/Other")
                original_text = f"{lang_info}\n\n{original_text}"
                
                # Second transcription forcing English translation
                self.chunk_progress.emit(f"Translating chunk {chunk_index} to English")
                result_english = model.transcribe(
                    chunk_file,
                    task="translate",
                    language=detected_language,
                    word_timestamps=True,
                    initial_prompt="Translate this multi-speaker conversation to standard English."
                )
                
                if not isinstance(result_english, dict) or "segments" not in result_english:
                    raise ValueError("Invalid translation result format")
                
                # Process translated segments with speaker diarization
                translated_speaker_segments = self.assign_speakers_to_transcript(
                    result_english.get("segments", []),  # Use get() with default empty list
                    speaker_segments
                )
                
                if not translated_speaker_segments:
                    raise ValueError("No valid segments found in translation")
                    
                translated_text = self.format_transcript(translated_speaker_segments)
                
                # Create results object
                chunk_result = {
                    'chunk_index': chunk_index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'original_text': original_text,
                    'translated_text': translated_text,
                    'original_segments': speaker_segments_with_text,
                    'translated_segments': translated_speaker_segments
                }
                
                # Create checkpoint
                self.create_checkpoint(chunk_index, chunk_result)
                
                return chunk_result
                
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_index}: {str(e)}"
                print(error_msg)
                self.error.emit(error_msg)
                return None
                
        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)
            return None
            
    def run(self):
        try:
            start_time = datetime.now()
            
            # Check if file exists first
            if not os.path.exists(self.audio_file):
                self.error.emit(f"Audio file not found: {self.audio_file}")
                return

            self.progress.emit(5)
            self.chunk_progress.emit("Loading Whisper model...")
            
            # Load model with FP32 for CPU compatibility
            model_load_start = datetime.now()
            model = whisper.load_model("medium", device="cpu")
            model_load_time = (datetime.now() - model_load_start).total_seconds()
            print(f"Model loading time: {model_load_time:.2f} seconds")
            self.progress.emit(10)
            
            # Check audio duration
            duration = self.get_audio_duration(self.audio_file)
            print(f"Audio duration: {duration:.2f} seconds")
            
            # For long audio files (longer than 20 minutes), use chunking
            if duration > 1200:  # 20 minutes threshold
                self.chunk_progress.emit(f"Long audio detected ({duration:.2f}s). Processing in chunks...")
                
                # Create temp directory for chunks
                temp_dir = os.path.join(os.path.dirname(self.audio_file), "temp_chunks")
                
                # Split audio into chunks
                self.chunk_progress.emit("Splitting audio into manageable chunks...")
                chunks = self.split_audio(self.audio_file, temp_dir)
                
                if not chunks:
                    self.error.emit("Failed to split audio file into chunks")
                    return
                    
                # Process each chunk
                chunk_results = []
                for i, chunk_info in enumerate(chunks):
                    if self._stop_requested:
                        self.error.emit("Processing stopped by user")
                        return
                        
                    chunk_file, start_time, end_time = chunk_info
                    self.chunk_progress.emit(f"Processing chunk {i+1}/{len(chunks)} [{self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}]")
                    
                    # Update progress (20-90% progress range for chunks)
                    progress_pct = 20 + int((i / len(chunks)) * 70)
                    self.progress.emit(progress_pct)
                    
                    # Process chunk
                    chunk_result = self.process_audio_chunk(chunk_info, model)
                    if chunk_result:
                        chunk_results.append(chunk_result)
                    
                # Merge results
                self.chunk_progress.emit("Merging results from all chunks...")
                self.progress.emit(95)
                original_text, translated_text, minutes, _ = self.merge_transcriptions(chunk_results)
                
                # Clean up temp files (optional)
                try:
                    for chunk_file, _, _ in chunks:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temporary files: {e}")
                
                self.progress.emit(100)
                self.finished.emit((original_text, translated_text, minutes))
                
            else:
                # For shorter files, use the original process
                self.chunk_progress.emit("Processing audio file normally...")
                
                # Get speaker diarization
                diarization_start = datetime.now()
                self.progress.emit(30)
                speaker_segments = self.get_speaker_diarization(self.audio_file)
                diarization_time = (datetime.now() - diarization_start).total_seconds()
                print(f"Speaker diarization time: {diarization_time:.2f} seconds")
                print(f"Speaker segments: {speaker_segments}")
                print(f"Proceeding to language detection...")
                
                if speaker_segments is None:
                    print("Falling back to basic speaker alternation")
                    speaker_segments = []  # Will trigger fallback in assign_speakers_to_transcript
                
                # First detect the language
                self.progress.emit(40)
                lang_detect_start = datetime.now()
                try:
                    initial_result = model.transcribe(
                        self.audio_file,
                        language=None,
                        word_timestamps=True,
                        task="transcribe"
                    )
                    
                    # Ensure initial_result is a dictionary
                    if not isinstance(initial_result, dict):
                        raise ValueError("Transcription result is not in the expected format")
                        
                    detected_language = initial_result.get("language", "en")
                    if not detected_language:
                        detected_language = "en"
                        
                    lang_detect_time = (datetime.now() - lang_detect_start).total_seconds()
                    print(f"Language detection time: {lang_detect_time:.2f} seconds")
                    print(f"Detected language: {detected_language}")
                    print(f"Proceeding to original transcription...")
                except Exception as e:
                    print(f"Language detection failed: {str(e)}, defaulting to English")
                    detected_language = "en"
                    lang_detect_time = 0.0
                
                # First transcription in original language (English/Tagalog)
                self.progress.emit(50)
                original_trans_start = datetime.now()
                # If detected language is not English or Tagalog, default to English
                if detected_language not in ["en", "tl"]:
                    detected_language = "en"
                    
                try:
                    result_original = model.transcribe(
                        self.audio_file,
                        task="transcribe",
                        language=detected_language,
                        word_timestamps=True,
                        initial_prompt="This audio may contain English and/or Tagalog language with multiple speakers."
                    )
                    
                    if not isinstance(result_original, dict) or "segments" not in result_original:
                        raise ValueError("Invalid transcription result format")
                        
                    original_trans_time = (datetime.now() - original_trans_start).total_seconds()
                    print(f"Original transcription time: {original_trans_time:.2f} seconds")
                    print(f"Proceeding to speaker assignment and Translation...")   
                    # Process segments with speaker diarization
                    speaker_assign_start = datetime.now()
                    speaker_segments_with_text = self.assign_speakers_to_transcript(
                        result_original.get("segments", []),  # Use get() with default empty list
                        speaker_segments
                    )
                    
                    if not speaker_segments_with_text:
                        raise ValueError("No valid segments found in transcription")
                        
                    original_text = self.format_transcript(speaker_segments_with_text)
                    speaker_assign_time = (datetime.now() - speaker_assign_start).total_seconds()
                    print(f"Speaker assignment time: {speaker_assign_time:.2f} seconds")
                    
                    # Add language detection info to original text
                    lang_info = "Detected Language: " + ("English" if detected_language == "en" else "Tagalog/English" if detected_language == "tl" else "Mixed/Other")
                    original_text = f"{lang_info}\n\n{original_text}"
                    
                    # Second transcription forcing English translation
                    self.progress.emit(75)
                    translation_start = datetime.now()
                    result_english = model.transcribe(
                        self.audio_file,
                        task="translate",
                        language=detected_language,
                        word_timestamps=True,
                        initial_prompt="Translate this multi-speaker conversation to standard English."
                    )
                    
                    if not isinstance(result_english, dict) or "segments" not in result_english:
                        raise ValueError("Invalid translation result format")
                        
                    translation_time = (datetime.now() - translation_start).total_seconds()
                    print(f"Translation time: {translation_time:.2f} seconds")
                    print(f"Proceeding to translated segments with speaker diarization...")
                    print(f"Proceeding to minutes generation...")
                    
                    # Process translated segments with speaker diarization
                    translated_speaker_segments = self.assign_speakers_to_transcript(
                        result_english.get("segments", []),  # Use get() with default empty list
                        speaker_segments
                    )
                    
                    if not translated_speaker_segments:
                        raise ValueError("No valid segments found in translation")
                        
                    translated_text = self.format_transcript(translated_speaker_segments)
                    
                    # Generate meeting minutes
                    minutes_start = datetime.now()
                    minutes = self.generate_meeting_minutes(speaker_segments_with_text)
                    minutes_time = (datetime.now() - minutes_start).total_seconds()
                    print(f"Minutes generation time: {minutes_time:.2f} seconds")
                    
                    # Calculate total time
                    total_time = (datetime.now() - start_time).total_seconds()
                    time_summary = f"\n\nProcessing Time Summary:\n"
                    time_summary += f"Model Loading: {model_load_time:.2f}s\n"
                    time_summary += f"Speaker Diarization: {diarization_time:.2f}s\n"
                    time_summary += f"Language Detection: {lang_detect_time:.2f}s\n"
                    time_summary += f"Original Transcription: {original_trans_time:.2f}s\n"
                    time_summary += f"Speaker Assignment: {speaker_assign_time:.2f}s\n"
                    time_summary += f"Translation: {translation_time:.2f}s\n"
                    time_summary += f"Minutes Generation: {minutes_time:.2f}s\n"
                    time_summary += f"Total Processing Time: {total_time:.2f}s"
                    
                    # Add time summary to minutes
                    minutes += time_summary
                    
                    self.progress.emit(100)
                    self.finished.emit((original_text, translated_text, minutes))
                    
                except Exception as e:
                    error_msg = f"Error during transcription: {str(e)}"
                    print(error_msg)
                    self.error.emit(error_msg)
                    
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)

class AudioTranscriberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Speaker Audio Transcriber (English/Tagalog)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize instance variables
        self.audio_file = None
        self.transcriber = None
        self.current_progress = 0
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create info label with better styling
        info_label = QLabel("This transcriber is optimized for English and Tagalog audio with multiple speakers")
        info_label.setStyleSheet("color: #2962FF; padding: 10px; background-color: #E3F2FD; border-radius: 5px;")
        main_layout.addWidget(info_label)
        
        # Add saving guidelines info
        save_info = QLabel(
            "Saving Files Guide:\n"
            "• All files are saved in the 'transcription_outputs' folder\n"
            "• Files are automatically named with timestamp and type (e.g., filename_transcription_20250307_092600.txt)\n"
            "• 'Save Complete Output' combines all sections with clear separators\n"
            "• Individual sections can be saved separately using their respective save buttons\n"
            "• The 'Save Minutes Only' button saves only the meeting minutes section\n"
            "• The 'Save Transcription' button saves only the Original transcription section\n"
            "• The 'Save Translation' button saves only the Translated transcription section\n"
            "• To completely save the output, use the 'Save Complete Output' button from the Meeting minutes section"
        )
        save_info.setStyleSheet("""
            QLabel {
                color: #424242;
                padding: 10px;
                background-color: #FFF3E0;
                border-radius: 5px;
                margin: 5px;
                font-size: 11px;
            }
        """)
        main_layout.addWidget(save_info)
        
        # Create control panel with better organization
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_panel)
        
        # Create top row for file selection
        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #424242;")
        self.load_button = QPushButton("Load Audio File")
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        file_row.addWidget(self.file_label)
        file_row.addWidget(self.load_button)
        control_layout.addLayout(file_row)
        
        # Create second row for speaker count and transcribe button
        action_row = QHBoxLayout()
        speaker_group = QGroupBox("Number of Speakers")
        speaker_layout = QHBoxLayout(speaker_group)
        self.speaker_count = QSpinBox()
        self.speaker_count.setRange(1, 10)
        self.speaker_count.setValue(2)
        self.speaker_count.setStyleSheet("padding: 3px;")
        speaker_layout.addWidget(self.speaker_count)
        action_row.addWidget(speaker_group)
        
        # Add chunk size control
        chunk_group = QGroupBox("Chunk Size (minutes)")
        chunk_layout = QHBoxLayout(chunk_group)
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(5, 30)
        self.chunk_size.setValue(10)
        self.chunk_size.setStyleSheet("padding: 3px;")
        chunk_layout.addWidget(self.chunk_size)
        action_row.addWidget(chunk_group)
        
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setStyleSheet("""
            QPushButton {
                background-color: #43A047;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        action_row.addWidget(self.transcribe_button)
        control_layout.addLayout(action_row)
        
        # Create status row
        status_row = QHBoxLayout()
        self.status_label = QLabel("Select an audio file to transcribe")
        self.status_label.setStyleSheet("color: #424242;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2962FF;
            }
        """)
        self.progress_bar.hide()
        status_row.addWidget(self.status_label)
        status_row.addWidget(self.progress_bar)
        control_layout.addLayout(status_row)
        
        # Add chunk progress label
        self.chunk_label = QLabel("")
        self.chunk_label.setStyleSheet("""
            QLabel {
                color: #1565C0;
                background-color: #E3F2FD;
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        self.chunk_label.hide()
        control_layout.addWidget(self.chunk_label)
        
        # Add chunk progress bar
        chunk_progress_row = QHBoxLayout()
        chunk_progress_label = QLabel("Chunk Progress:")
        chunk_progress_label.setStyleSheet("color: #424242;")
        self.chunk_progress_bar = QProgressBar()
        self.chunk_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
            }
        """)
        self.chunk_progress_bar.hide()
        chunk_progress_row.addWidget(chunk_progress_label)
        chunk_progress_row.addWidget(self.chunk_progress_bar)
        control_layout.addLayout(chunk_progress_row)
        
        # Add current chunk indicator
        self.chunk_counter_label = QLabel("")
        self.chunk_counter_label.setStyleSheet("""
            QLabel {
                color: #FF5722;
                padding: 5px;
                font-weight: bold;
            }
        """)
        self.chunk_counter_label.hide()
        control_layout.addWidget(self.chunk_counter_label)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Create tab widget for output with better styling
        output_tabs = QTabWidget()
        output_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #F5F5F5;
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2962FF;
                color: white;
            }
        """)
        
        # Original transcription tab
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        original_button_layout = QHBoxLayout()
        self.save_original_button = QPushButton("Save Transcription")
        self.save_original_button.clicked.connect(lambda: self.save_output("transcription"))
        self.save_original_button.setEnabled(False)
        original_button_layout.addWidget(self.save_original_button)
        self.original_output = QTextEdit()
        self.original_output.setReadOnly(True)
        self.original_output.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', monospace;
            }
        """)
        original_layout.addWidget(self.original_output)
        original_layout.addLayout(original_button_layout)
        output_tabs.addTab(original_tab, "Original Transcription")
        
        # Translation tab
        translation_tab = QWidget()
        translation_layout = QVBoxLayout(translation_tab)
        translation_button_layout = QHBoxLayout()
        self.save_translation_button = QPushButton("Save Translation")
        self.save_translation_button.clicked.connect(lambda: self.save_output("translation"))
        self.save_translation_button.setEnabled(False)
        translation_button_layout.addWidget(self.save_translation_button)
        self.translation_output = QTextEdit()
        self.translation_output.setReadOnly(True)
        self.translation_output.setStyleSheet(self.original_output.styleSheet())
        translation_layout.addWidget(self.translation_output)
        translation_layout.addLayout(translation_button_layout)
        output_tabs.addTab(translation_tab, "English Translation")
        
        # Minutes tab
        minutes_tab = QWidget()
        minutes_layout = QVBoxLayout(minutes_tab)
        minutes_button_layout = QHBoxLayout()
        self.save_minutes_button = QPushButton("Save Minutes Only")
        self.save_minutes_button.clicked.connect(lambda: self.save_output("minutes"))
        self.save_minutes_button.setEnabled(False)
        self.save_complete_button = QPushButton("Save Complete Output")
        self.save_complete_button.clicked.connect(lambda: self.save_output("complete"))
        self.save_complete_button.setEnabled(False)
        minutes_button_layout.addWidget(self.save_minutes_button)
        minutes_button_layout.addWidget(self.save_complete_button)
        self.minutes_output = QTextEdit()
        self.minutes_output.setReadOnly(True)
        self.minutes_output.setStyleSheet(self.original_output.styleSheet())
        minutes_layout.addWidget(self.minutes_output)
        minutes_layout.addLayout(minutes_button_layout)
        output_tabs.addTab(minutes_tab, "Meeting Minutes")
        
        # Add output tabs to main layout
        main_layout.addWidget(output_tabs)
        
        # Connect signals
        self.load_button.clicked.connect(self.load_audio_file)
        self.transcribe_button.clicked.connect(self.start_transcription)
        
    def load_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg);;All Files (*.*)"
        )
        
        if file_name:
            try:
                if not os.path.exists(file_name):
                    self.status_label.setText("Error: Selected file does not exist")
                    return
                    
                # Check if file is readable
                with open(file_name, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
                    
                self.audio_file = file_name
                self.file_label.setText(f"Selected: {os.path.basename(file_name)}")
                self.transcribe_button.setEnabled(True)
                self.status_label.setText("Ready to transcribe")
                self.clear_outputs()
                
            except Exception as e:
                self.status_label.setText(f"Error: Could not read file - {str(e)}")
                
    def clear_outputs(self):
        """Clear all output text areas and disable save buttons"""
        self.original_output.clear()
        self.translation_output.clear()
        self.minutes_output.clear()
        self.save_original_button.setEnabled(False)
        self.save_translation_button.setEnabled(False)
        self.save_minutes_button.setEnabled(False)
        self.save_complete_button.setEnabled(False)
        self.chunk_label.hide()
        self.chunk_progress_bar.hide()
        self.chunk_counter_label.hide()
        
    def start_transcription(self):
        if not self.audio_file:
            return
            
        # Update UI state
        self.status_label.setText("Transcribing...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.chunk_label.setText("Initializing...")
        self.chunk_label.show()
        self.chunk_progress_bar.setValue(0)
        self.chunk_progress_bar.show()
        self.chunk_counter_label.hide()
        self.transcribe_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.clear_outputs()
        
        # Store chunk processing info
        self.total_chunks = 0
        self.current_chunk = 0
        
        # Stop any existing transcription
        if self.transcriber is not None:
            self.transcriber.stop()
            self.transcriber.wait()
            
        # Start new transcription
        chunk_size_minutes = self.chunk_size.value()
        self.transcriber = TranscriptionThread(
            self.audio_file, 
            self.speaker_count.value(),
            chunk_size_seconds=chunk_size_minutes * 60
        )
        self.transcriber.finished.connect(self.handle_transcription_finished)
        self.transcriber.error.connect(self.handle_transcription_error)
        self.transcriber.progress.connect(self.update_progress)
        self.transcriber.chunk_progress.connect(self.update_chunk_progress)
        self.transcriber.start()
        
    def update_progress(self, value):
        """Smoothly update progress bar"""
        if value > self.current_progress:
            self.current_progress = value
            self.progress_bar.setValue(value)
            
    def update_chunk_progress(self, message):
        """Update chunk processing status with detailed tracking"""
        self.chunk_label.setText(message)
        
        # Extract chunk information if available
        if "Splitting audio: chunk" in message:
            try:
                # Parse "Splitting audio: chunk X/Y"
                parts = message.split()
                fraction = parts[3].split("/")
                current = int(fraction[0])
                total = int(fraction[1])
                self.total_chunks = total
                
                # Update chunk progress bar for splitting phase
                self.chunk_progress_bar.setValue(int(current / total * 100))
                self.chunk_counter_label.setText(f"Splitting: {current}/{total}")
                self.chunk_counter_label.show()
            except:
                pass
        
        elif "Processing chunk" in message:
            try:
                # Parse "Processing chunk X/Y"
                parts = message.split()
                fraction = parts[2].split("/")
                current = int(fraction[0])
                total = int(fraction[1])
                self.current_chunk = current
                self.total_chunks = total
                
                # Update chunk progress bar
                self.chunk_progress_bar.setValue(int(current / total * 100))
                self.chunk_counter_label.setText(f"Processing chunk: {current}/{total}")
                self.chunk_counter_label.show()
            except:
                pass
                
        elif "Diarizing speakers for chunk" in message or "Detecting language for chunk" in message:
            try:
                # Parse the chunk index
                parts = message.split()
                chunk_index = int(parts[-1])
                
                # Update chunk progress bar for diarization or language detection phase (25% of this chunk's progress)
                self.chunk_progress_bar.setValue(25)
            except:
                pass
                
        elif "Transcribing chunk" in message:
            try:
                # Parse the chunk index
                parts = message.split()
                chunk_index = int(parts[-4])
                
                # Update chunk progress bar for transcription phase (50% of this chunk's progress)
                self.chunk_progress_bar.setValue(50)
            except:
                pass
                
        elif "Translating chunk" in message:
            try:
                # Parse the chunk index
                parts = message.split()
                chunk_index = int(parts[-2])
                
                # Update chunk progress bar for translation phase (75% of this chunk's progress)
                self.chunk_progress_bar.setValue(75)
            except:
                pass
                
        elif "Loaded checkpoint for chunk" in message:
            try:
                # Parse the chunk index
                parts = message.split()
                chunk_index = int(parts[-1])
                
                # Update chunk progress bar for loaded checkpoint (100% for this chunk)
                self.chunk_progress_bar.setValue(100)
            except:
                pass
                
        elif "Merging results" in message:
            # When merging, show 100% on chunk progress
            self.chunk_progress_bar.setValue(100)
            self.chunk_counter_label.setText("Merging all chunks...")
        
    def handle_transcription_finished(self, results):
        """Handle transcription results with error checking"""
        try:
            if not results or len(results) != 3:
                raise ValueError("Invalid transcription results")
                
            original_text, translated_text, minutes = results
            
            # Update UI with results
            self.status_label.setText("Transcription complete")
            self.progress_bar.hide()
            self.chunk_label.hide()
            self.chunk_progress_bar.hide()
            self.chunk_counter_label.hide()
            self.original_output.setText(original_text)
            self.translation_output.setText(translated_text)
            self.minutes_output.setText(minutes)
            
            # Enable save buttons
            self.save_original_button.setEnabled(True)
            self.save_translation_button.setEnabled(True)
            self.save_minutes_button.setEnabled(True)
            self.save_complete_button.setEnabled(True)
            
        except Exception as e:
            self.handle_transcription_error(f"Error processing results: {str(e)}")
            
        finally:
            self.transcribe_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.current_progress = 0
            
    def handle_transcription_error(self, error_message):
        """Handle transcription errors with user-friendly messages"""
        self.status_label.setText("Error occurred")
        self.progress_bar.hide()
        self.chunk_progress_bar.hide()
        self.chunk_counter_label.hide()
        self.chunk_label.hide()
        
        # Show error in all outputs for visibility
        error_text = f"Error: {error_message}\n\nPlease try again or check the following:\n" \
                    "1. Audio file is not corrupted\n" \
                    "2. Internet connection is stable\n" \
                    "3. Sufficient system memory is available"
        
        self.original_output.setText(error_text)
        self.translation_output.setText(error_text)
        self.minutes_output.setText(error_text)
        
        # Reset UI state
        self.transcribe_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.current_progress = 0

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if self.transcriber is not None:
            self.transcriber.stop()
            self.transcriber.wait()
        event.accept()

    def save_output(self, output_type):
        """Save the selected output to a text file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = "transcription_outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(self.audio_file))[0]
            
            # Initialize content and filename
            content = ""
            filename = ""
            
            if output_type == "complete":  # Changed from "minutes" to "complete"
                # Save combined output (all sections)
                filename = f"{base_filename}_complete_output_{timestamp}.txt"
                
                # Prepare content with clear section separators
                content = "=" * 80 + "\n"
                content += "ORIGINAL TRANSCRIPTION\n"
                content += "=" * 80 + "\n\n"
                content += self.original_output.toPlainText() + "\n\n"
                
                content += "=" * 80 + "\n"
                content += "ENGLISH TRANSLATION\n"
                content += "=" * 80 + "\n\n"
                content += self.translation_output.toPlainText() + "\n\n"
                
                content += "=" * 80 + "\n"
                content += "MEETING MINUTES\n"
                content += "=" * 80 + "\n\n"
                content += self.minutes_output.toPlainText() + "\n"
                
                # Add metadata at the end
                content += "\n" + "=" * 80 + "\n"
                content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                content += f"Audio file: {self.audio_file}\n"
                content += f"Number of speakers: {self.speaker_count.value()}\n"
                
            elif output_type == "transcription":
                content = self.original_output.toPlainText()
                filename = f"{base_filename}_transcription_{timestamp}.txt"
            elif output_type == "translation":
                content = self.translation_output.toPlainText()
                filename = f"{base_filename}_translation_{timestamp}.txt"
            elif output_type == "minutes":
                content = self.minutes_output.toPlainText()
                filename = f"{base_filename}_minutes_{timestamp}.txt"
            
            # Save the file
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Show success message
            self.status_label.setText(f"Saved to: {filename}")
            
        except Exception as e:
            self.status_label.setText(f"Error saving output: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 