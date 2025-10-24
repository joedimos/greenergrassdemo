import threading
import queue
import os
import platform
import subprocess
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


class LiveCallAIAgent:
    def __init__(self, api_key=None, tts_method='system'):
        """
        Initialize the AI agent with transcription and response capabilities.
        
        Args:
            api_key: Mistral API key
            tts_method: 'system' or 'mistral' (for TTS)
        """
        
        # Get Mistral API key
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
             raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        
        # Audio settings
        self.SAMPLE_RATE = 16000  # Whisper works best at 16kHz
        self.CHANNELS = 1
        self.RECORD_SECONDS = 20  # Record in 11-second chunks
        
        # Determine TTS method
        self.tts_method = tts_method
        print(f"Using TTS method: {self.tts_method}")
        
        # Initialize LangChain with Mistral
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.9,
            mistral_api_key=self.api_key
        )
        
        # Create custom prompt template
        template = """You are a helpful AI assistant on a phone call. You provide clear, 
        concise, and logical responses. Keep your answers brief and conversational, 
        as this is a live phone conversation. Aim for responses under 3 sentences unless 
        more detail is specifically requested.

        Current conversation:
        {history}
        Human: {input}
        AI Assistant:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        # Initialize conversation memory and chain
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )
        
        # Threading and queue for async processing
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.transcript_log = []
        
        # Create temp directory for audio files
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Test audio device
        print("\n🔊 Available audio devices:")
        print(sd.query_devices())
        print("\n✓ Agent initialized successfully")
    
    def record_audio_chunk(self):
        """Record a chunk of audio using sounddevice."""
        try:
            print("🎤 Recording...")
            recording = sd.rec(
                int(self.RECORD_SECONDS * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='int16'
            )
            sd.wait()  # Wait until recording is finished
            return recording
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def save_audio_to_file(self, recording, filename):
        """Save recorded audio to WAV file."""
        write_wav(str(filename), self.SAMPLE_RATE, recording)
    
    def transcribe_audio_whisper(self, audio_file):
        """Transcribe audio using local Whisper model."""
        try:
            # Import whisper locally to handle different installation methods
            try:
                import whisper
                # Load model once and cache it
                if not hasattr(self, 'whisper_model'):
                    print("Loading Whisper model (first time only)...")
                    self.whisper_model = whisper.load_model("base")
                
                result = self.whisper_model.transcribe(str(audio_file))
                text = result["text"].strip()
                return text if len(text) > 0 else None
            except (ImportError, AttributeError) as e:
                print(f"❌ Whisper import error: {e}")
                print("\nPlease install whisper correctly:")
                print("pip uninstall whisper openai-whisper -y")
                print("pip install -U openai-whisper")
                return None
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def speak(self, text):
        """Convert text to speech using available method."""
        print(f"\n🤖 AI: {text}\n")
        
        try:
            if self.tts_method == 'system':
                self._speak_system(text)
            elif self.tts_method == 'mistral':
                # Mistral doesn't have native TTS, use system as fallback
                print("ℹ️  Mistral doesn't have native TTS, using system TTS")
                self._speak_system(text)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def _speak_system(self, text):
        """Use system TTS command."""
        system = platform.system()
        try:
            if system == 'Darwin':  # macOS
                subprocess.run(['say', text], check=True)
            elif system == 'Windows':
                # Escape quotes for PowerShell
                text_escaped = text.replace('"', '`"')
                ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text_escaped}")'
                subprocess.run(['powershell', '-Command', ps_command], capture_output=True, check=True)
            elif system == 'Linux':
                subprocess.run(['espeak', text], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"System TTS not available: {e}")
    
    def _speak_openai(self, text):
        """Use third-party TTS API (keeping for compatibility)."""
        temp_file = self.temp_dir / f"tts_{datetime.now().timestamp()}.mp3"
        
        # This would require a separate TTS service
        # For Mistral setup, we recommend using system TTS or gTTS
        print("ℹ️  Using system TTS as fallback")
        self._speak_system(text)
    
    def process_with_langchain(self, user_input):
        """Process user input through LangChain and get AI response."""
        try:
            response = self.conversation.predict(input=user_input)
            return response
        except Exception as e:
            print(f"Error processing with LangChain: {e}")
            return "I apologize, but I encountered an error. Could you please repeat that?"
    
    def listen_worker(self):
        """Worker thread to continuously listen for audio."""
        print("🎤 Listening thread started...")
        
        while self.is_running:
            try:
                # Record audio chunk
                recording = self.record_audio_chunk()
                
                if recording is not None and self.is_running:
                    # Check if recording has enough volume (not silence)
                    volume = np.abs(recording).mean()
                    if volume > 50:  # Threshold for silence detection
                        # Save to temporary file
                        temp_audio = self.temp_dir / f"recording_{datetime.now().timestamp()}.wav"
                        self.save_audio_to_file(recording, temp_audio)
                        self.audio_queue.put(temp_audio)
                    else:
                        print("(Silence detected, skipping)")
                    
            except Exception as e:
                print(f"Error in listen worker: {e}")
                if not self.is_running:
                    break
    
    def process_worker(self):
        """Worker thread to process audio and generate responses."""
        print("🧠 Processing thread started...")
        
        while self.is_running:
            try:
                # Get audio file from queue
                audio_file = self.audio_queue.get(timeout=1)
                
                # Transcribe
                print("Transcribing...")
                text = self.transcribe_audio_whisper(audio_file)
                
                # Clean up audio file
                if audio_file.exists():
                    audio_file.unlink()
                
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n👤 User [{timestamp}]: {text}")
                    
                    # Log transcript
                    self.transcript_log.append({
                        'timestamp': timestamp,
                        'speaker': 'User',
                        'text': text
                    })
                    
                    # Check for exit commands
                    if any(phrase in text.lower() for phrase in ['goodbye', 'end call', 'hang up', 'stop listening', 'exit']):
                        response = "Thank you for calling. Goodbye!"
                        self.speak(response)
                        self.is_running = False
                        break
                    
                    # Process with LangChain
                    print("Generating response...")
                    response = self.process_with_langchain(text)
                    
                    # Log AI response
                    self.transcript_log.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'speaker': 'AI',
                        'text': response
                    })
                    
                    # Speak response
                    self.speak(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"Error in process worker: {e}")
    
    def start_call(self):
        """Start the live call session."""
        print("\n" + "="*60)
        print("🤖 LIVE CALL AI AGENT STARTED")
        print("="*60)
        print("Say 'goodbye', 'end call', or 'hang up' to end the session.")
        print("Press Ctrl+C to force quit.")
        print("="*60 + "\n")
        
        self.is_running = True
        
        # Start worker threads
        listen_thread = threading.Thread(target=self.listen_worker, daemon=True)
        process_thread = threading.Thread(target=self.process_worker, daemon=True)
        
        listen_thread.start()
        process_thread.start()
        
        # Initial greeting
        greeting = "Hello! I'm your AI assistant. How can I help you today?"
        self.speak(greeting)
        
        # Keep main thread alive
        try:
            while self.is_running:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            print("\n\nCall interrupted by user.")
            self.is_running = False
        
        # Wait for threads to finish
        listen_thread.join(timeout=5)
        process_thread.join(timeout=5)
        
        print("\n" + "="*60)
        print("🤖 CALL ENDED")
        print("="*60)
        
        # Cleanup
        self.save_transcript()
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            for file in self.temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except:
                    pass
            for file in self.temp_dir.glob("*.mp3"):
                try:
                    file.unlink()
                except:
                    pass
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def save_transcript(self):
        """Save the call transcript to a file."""
        if not self.transcript_log:
            print("No transcript to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"call_transcript_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("CALL TRANSCRIPT\n")
            f.write("="*60 + "\n\n")
            
            for entry in self.transcript_log:
                f.write(f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}\n\n")
        
        print(f"\n📄 Transcript saved to: {filename}")


# Example usage
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         LIVE PHONE CALL AI TRANSCRIPTION AGENT            ║
    ║              Powered by LangChain & Mistral               ║
    ╚════════════════════════════════════════════════════════════╝
    
    EASY INSTALLATION:
    pip install langchain langchain-mistralai mistralai sounddevice scipy numpy openai-whisper
    
    Prerequisites:
    1. Set MISTRAL_API_KEY environment variable
    2. Ensure microphone permissions are enabled
   
    """)
    
    try:
        # Initialize agent with system TTS
        agent = LiveCallAIAgent(tts_method='system')
        
        agent.start_call()
        
    except Exception as e:
        print(f"\n❌ Error starting agent: {e}")
        print("\nTroubleshooting:")
        print("1. Set API key: export MISTRAL_API_KEY='your-key'")
        print("2. Install packages: pip install langchain langchain-mistralai mistralai sounddevice scipy numpy openai-whisper")
        print("3. Check microphone permissions in system settings")
        print("4. On macOS, you may need to allow terminal to access microphone")
