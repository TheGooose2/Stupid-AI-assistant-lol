import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import speech_recognition as sr

# Load DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('voice', engine.getProperty('voices')[0].id)

# Memory for context
chat_history_ids = None

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            print("Timeout waiting for speech.")
            return ""
    try:
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        return query
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError:
        speak("Speech recognition is down right now.")
        return ""

def generate_response(prompt, chat_history_ids):
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8
    )
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip(), chat_history_ids

def main():
    global chat_history_ids
    speak("Jarvis online. What’s good, Blake?")
    while True:
        command = listen()
        if not command:
            continue
        if any(word in command.lower() for word in ["exit", "shutdown", "stop"]):
            speak("Shutting down. Later king.")
            break
        response, chat_history_ids = generate_response(command, chat_history_ids)
        speak(response)

if __name__ == "__main__":
    main()
