import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()  # Initialize client with API key from env automatically

def test_whisper():
    audio_file_path = "test_audio.mp3"

    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
        print("Whisper transcript:", transcript.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_whisper()
