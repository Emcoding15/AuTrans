# Multi-Speaker Audio Transcriber

A powerful desktop application for transcribing audio files with multiple speakers, supporting both English and Tagalog languages. This tool provides real-time transcription, translation, and meeting minutes generation.

## Features

- 🎯 Multi-speaker detection and diarization
- 🌐 Support for English and Tagalog languages
- 🔄 Automatic translation to English
- 📝 Meeting minutes generation
- 📊 Progress tracking with detailed status updates
- 💾 Checkpoint system for long recordings
- 📁 Multiple output formats and saving options
- 🎨 User-friendly GUI interface

## Prerequisites

Before running the application, make sure you have:

1. Python 3.8 or higher installed
2. HuggingFace API token (for speaker diarization)
3. Ollama installed and running locally (for meeting minutes generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AudioTranscriber.git
cd AudioTranscriber
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your HuggingFace API token:
   - Create an account at [HuggingFace](https://huggingface.co)
   - Get your API token from your account settings
   - Set it as an environment variable:
     ```bash
     # Windows
     set HUGGINGFACE_TOKEN=your_token_here
     
     # Linux/Mac
     export HUGGINGFACE_TOKEN=your_token_here
     ```

4. Install and start Ollama:
   - Download from [Ollama's website](https://ollama.ai)
   - Install and start the Ollama service
   - Pull the required model:
     ```bash
     ollama pull llama2
     ```

## Usage

1. Start the application:
```bash
python audio_transcriber.py
```

2. Using the GUI:
   - Click "Load Audio File" to select your audio file (supports .mp3, .wav, .m4a, .ogg)
   - Set the number of speakers (1-10)
   - Adjust chunk size for long recordings (5-30 minutes)
   - Click "Transcribe" to start the process

3. Monitor Progress:
   - Watch the progress bar for overall completion
   - Check the chunk progress for detailed status
   - View real-time updates in the status area

4. View Results:
   - Original Transcription tab: Shows the transcription in the original language
   - English Translation tab: Shows the English translation
   - Meeting Minutes tab: Shows the generated meeting minutes

5. Save Results:
   - Save individual sections using the respective buttons
   - Use "Save Complete Output" to save all sections in one file
   - Files are saved in the `transcription_outputs` folder with timestamps

## Output Files

The application generates several types of output files:

1. Complete Output (`filename_complete_output_TIMESTAMP.txt`):
   - Contains all sections (original, translation, and minutes)
   - Includes metadata and processing information

2. Individual Sections:
   - Original Transcription (`filename_transcription_TIMESTAMP.txt`)
   - English Translation (`filename_translation_TIMESTAMP.txt`)
   - Meeting Minutes (`filename_minutes_TIMESTAMP.txt`)

## Tips for Best Results

1. Audio Quality:
   - Use clear audio recordings with minimal background noise
   - Ensure speakers are well-separated and clearly audible

2. Processing Long Files:
   - For recordings longer than 20 minutes, adjust the chunk size
   - Larger chunks (20-30 minutes) are better for maintaining context
   - Smaller chunks (5-10 minutes) are better for memory management

3. Speaker Count:
   - Set the correct number of speakers for better diarization
   - If unsure, start with 2 speakers and adjust as needed

## Troubleshooting

1. If the application fails to start:
   - Check if Python and all dependencies are installed correctly
   - Verify your HuggingFace token is set correctly
   - Ensure Ollama is running and the llama2 model is installed

2. If transcription fails:
   - Check your internet connection
   - Verify the audio file is not corrupted
   - Ensure sufficient system memory is available

3. If speaker diarization is inaccurate:
   - Adjust the number of speakers
   - Try processing in smaller chunks
   - Check audio quality and speaker separation

## Support

For issues and feature requests, please create an issue in the GitHub repository.

## License

[Your chosen license]

## Acknowledgments

- Whisper for speech recognition
- Pyannote for speaker diarization
- SpeechBrain for speaker embeddings
- Ollama for meeting minutes generation 