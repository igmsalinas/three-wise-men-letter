# Import necessary libraries
from nicegui import ui   
import subprocess        
import tempfile          
import wave              
import os                
import numpy as np       
import speech_recognition as sr  
from datetime import datetime  
from queue import Queue  
from time import sleep   
import threading
from silero_vad import load_silero_vad, get_speech_timestamps
import google.generativeai as genai
import dotenv
import warnings

warnings.filterwarnings('ignore')

# Load the Gemini API KEY
dotenv.load_dotenv('.env')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Global variables
recording = False
transcription = ['']
data_queue = Queue()
recorder = sr.Recognizer()
RATE = 16000
vad = load_silero_vad()
previous_audio = np.zeros([], dtype=np.int16)
transcribe_time = 0
thread = None
text = None

# Global source variable, microphone initialized once
source = None
stop_listening = None

# Function to initialize the microphone source
def init_microphone():
    global source
    global recorder
    if source is None:
        source = sr.Microphone(sample_rate=RATE)
        with source:
            recorder.adjust_for_ambient_noise(source)
    return source

# Function to start recording
def start_recording():
    global recording, transcription, thread, stop_listening
    recording = True
    transcription = ['']
    ui.notify('Recording started', type='info')
    print(f"{datetime.now().time()} | [INFO]: START RECORDING")

    source = init_microphone()

    def record_callback(_, audio: sr.AudioData):
        data_queue.put(audio.get_raw_data())

    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=2)

    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=transcription_worker)
        thread.start()

# Function to stop recording
def stop_recording():
    global recording, stop_listening
    recording = False
    stop_listening(wait_for_stop=False)
    ui.notify('Recording stopped', type='info')
    print(f"{datetime.now().time()} | [INFO]: STOP RECORDING")

# Function to transcribe audio to text
def transcribe_to_txt(input_filename: str, output_filename: str):
    command = ['./whisper.cpp/main', '-f', input_filename, '-otxt', '-of', output_filename, '-m', './whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin', '-l', 'auto']
    subprocess.run(command, capture_output=True, text=True)

# Modified generate_summary function to hide the image and record button
def generate_summary():
    response = model.generate_content(
        f"Genera un lista de los objetos deseados como regalo en formato carta para los Reyes Magos, "
        f"dependiendo del texto a continuación. Utiliza un lenguaje de género neutro. Si no recibes datos simplemente responde con 'Carta no disponible, "
        f"algo ha ido mal :(' . La carta debe empezar con un título que sea \n #Carta a los Reyes Magos\n Deja una línea en blanco y continúa con: "
        f"\n Queridos Reyes Magos: \n (Introducción diciendo que me he portado muy bien... etc) \n "
        f"Este año quiero: \n Bullets de lista de cosas que quiere una por una, dando un detalle breve y por qué lo quiere. Incluye TODAS las cosas que quiere "
        f" \n Da las gracias y espera que les guste la carta. \n Atentamente, \n"
        f"\n*Texto*:\n '{' '.join(transcription)}'"
    )
    try:
        summary = response.text
        output_display.set_content(f"{summary}")
        output_display.style("padding-bottom: 80px")
        print(f"{datetime.now().time()} | [INFO]: GENERATED SUMMARY")
        
        # Hide the image and the record button
        banner_image.style('display: none;')
        record_button.style('display: none;')

    except Exception as e:
        print(f"{datetime.now().time()} | [ERROR]: {e}")
        output_display.set_content("Carta no disponible algo ha ido mal :(")

# Background worker for recording and transcription
def transcription_worker():
    global transcription, transcribe_time, previous_audio, recording, data_queue, text
    print(f"{datetime.now().time()} | [INFO]: TRANSCRIBE THREAD STARTED")

    while recording:
        if not data_queue.empty():
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            if len(get_speech_timestamps(audio_np, vad)) > 0:
                previous_audio = np.append(previous_audio, audio_np)
                speech_timestamps = get_speech_timestamps(previous_audio, vad)
                
                if len(speech_timestamps) > 0:
                    print(f"{datetime.now().time()} | [INFO]: VOICE DETECTED")
                    end = speech_timestamps[-1]["end"]
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', prefix='audio_', dir='.') as tmpfile:
                        with wave.open(tmpfile.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(16000)
                            wav_file.writeframes(previous_audio[:end])

                        output_filename = tmpfile.name.replace('.wav', '')
                        transcribe_to_txt(tmpfile.name, output_filename)

                        with open(output_filename + '.txt', 'r') as file:
                            text = file.read()
                        os.remove(output_filename + '.txt')

                    transcribe_time += 2

                    if transcribe_time >= 30:
                        transcription[-1] = text
                        transcription.append('')
                        transcribe_time = 0
                        previous_audio = np.zeros([], dtype=np.int16)
                    else:
                        transcription[-1] = text
            else:
                if transcription[-1] != '':
                    transcription.append('')
                transcribe_time = 0
                previous_audio = np.zeros([], dtype=np.int16)
            update_display()
        sleep(0.1)
    print(f"{datetime.now().time()} | [INFO]: TRANSCRIBE THREAD FINISHED")
    generate_summary()

# Function to update the UI with transcription
def update_display():
    transcription_text = '\n'.join(transcription)
    output_display.set_content(transcription_text)

# ToggleButton class to control recording
class ToggleButton(ui.button):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state = False
        self.on('click', self.toggle)
        self.update()

    def toggle(self) -> None:
        if self._state:
            stop_recording()
        else:
            start_recording()
        self._state = not self._state
        self.update()

    def update(self) -> None:
        if self._state:
            self.props('color=red')
            self.set_text('Terminar Grabación')
            self.set_icon('stop')
        else:
            self.props('color=green')
            self.set_text('Empezar Grabación')
            self.set_icon('mic')
        super().update()

# Function to reset the UI to its initial state
def reset_ui():
    global transcription, recording, thread
    transcription = ['']  # Reset transcription
    recording = False
    output_display.set_content("**El texto irá apareciendo aquí ✍🏼...**")  # Reset display
    # Show image and button again
    banner_image.style('display: block;')
    record_button.style('display: block;')
    
    # Reset the ToggleButton state (this is handled by the ToggleButton class itself)
    if record_button._state:
        record_button.toggle()

# Function to build the UI
def build_ui():
    ui.query('body').style(f'background-image: url("https://cdn.pixabay.com/photo/2017/10/26/19/45/red-2892235_1280.png"); background-size: cover;')
    
    with ui.column().classes('w-full').style('margin: 50px auto; max-width: 70%; text-align: center; gap: 30px display: flex; align-items: center;'):
        # Assign ID to the image for easy access
        global banner_image
        banner_image = ui.image("./static/banner.jpg").style("border-radius: 15px;")
        
        # Assign ID to the ToggleButton instance
        global record_button
        record_button = ToggleButton('Empezar Grabación').props('icon-position=left').style(
            'margin-top: 20px; color: white; padding: 10px; background-color: #4CAF50; border-radius: 10px;'
        ).classes('hover:bg-green-600')
        
        # Output display for the transcription and summary
        global output_display
        output_display = ui.markdown('**El texto irá apareciendo aquí ✍🏼...**').style(
            'max-width: 100%; padding: 20px 30px; background-color: #FFF8E1; border-radius: 10px; '
            'box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); font-family: "Patrick Hand", cursive; '
            'font-size: 18px; color: #3e2723; text-align: left; width: 100%; margin: 20px'
        ).classes('w-full')

# Run the app
@ui.page('/',title='Title')
async def root():
    build_ui()

ui.run(reload=False)
