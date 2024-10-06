# Working with a bug of visualization
import sounddevice as sd
import numpy as np
import whisper
import torch
import warnings
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the Whisper model
whisper_model = whisper.load_model("base")

# Parameters for recording
samplerate = 16000  # Whisper works best with 16kHz audio
duration = 7  # Record for 7 seconds

# Function to record audio
def record_audio(duration, samplerate):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()  # Wait until the recording is finished
    return np.squeeze(audio)

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio):
    result = whisper_model.transcribe(audio)
    return result['text']

# Function to interact with PandasAI based on transcribed input
def query_pandasai(transcription):
    # Load your CSV file into a Pandas DataFrame
    df = pd.read_csv("loan.csv")  # Replace with the correct path to your CSV file
    
    # Set up the LLM and SmartDataFrame for PandasAI
    llm = OpenAI(api_token="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")  # Use your own OpenAI API key here
    sdf = SmartDataframe(df, config={"llm": llm})
    
    # Pass the transcription as a query to PandasAI
    response = sdf.chat(transcription)
    
    # Print the PandasAI response
    print(f"Response from PandasAI: {response}")

# Main function
def main():
    # Step 1: Record audio
    audio = record_audio(duration, samplerate)

    # Step 2: Transcribe the audio using Whisper
    transcription = transcribe_audio(audio)
    print(f"Transcription: {transcription}")

    # Step 3: Pass the transcription to PandasAI for analysis
    query_pandasai(transcription)

if __name__ == "__main__":
    main()
