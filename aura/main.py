
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time
import re

import requests
import sounddevice as sd
import pyttsx3

from faster_whisper import WhisperModel


# ==================================================
# AURA CONFIGURATION
# ==================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
AI_MODEL = "llama3.2:1b"

SAMPLE_RATE = 16000
RECORD_SECONDS = 6
WAKE_SECONDS = 3

WHISPER_MODEL = "base"

SYSTEM_PROMPT = """
You are Aura, a friendly and intelligent AI assistant.

IMPORTANT RULES:

1. Communicate ONLY in English.
2. Understand and respond to questions in English.
3. Give useful, accurate and relevant answers.
4. Never reply in Tamil, Hindi or any other language.
5. Keep your answers simple, natural and conversational.
6. Understand the user's question before answering.
7. If you do not understand, politely ask the user to repeat.
8. Do not make up information.
"""


# ==================================================
# GLOBAL VARIABLES
# ==================================================

whisper_model = None

whisper_lock = threading.Lock()
mic_lock = threading.Lock()
tts_lock = threading.Lock()
conversation_lock = threading.Lock()
state_lock = threading.Lock()

conversation_history = []


# ==================================================
# LOAD WHISPER MODEL
# ==================================================

def get_whisper_model():

    global whisper_model

    if whisper_model is None:

        print("Loading Whisper model...")

        whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print("Whisper model loaded.")

    return whisper_model


# ==================================================
# ENGLISH SPEECH RECOGNITION
# ==================================================

def transcribe_audio(audio):

    try:

        with whisper_lock:

            model = get_whisper_model()

            segments, info = model.transcribe(
                audio.flatten(),
                language="en",
                beam_size=3,
                vad_filter=True,
                task="transcribe",
                condition_on_previous_text=False
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            print("Recognized English:", text)

            return text

    except Exception as error:

        print("Transcription error:", error)

        return ""


# ==================================================
# WAKE WORD DETECTION
# ==================================================

def detect_wake_word(audio):

    try:

        with whisper_lock:

            model = get_whisper_model()

            segments, info = model.transcribe(
                audio.flatten(),
                language="en",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt="Hey Aura."
            )

            detected_text = " ".join(
                segment.text
                for segment in segments
            ).lower()

        detected_text = re.sub(
            r"[^a-z ]",
            "",
            detected_text
        )

        print("Wake word test:", detected_text)

        wake_words = [
            "hey aura",
            "hay aura",
            "hey oura",
            "hey ora",
            "hey ahura",
            "hey laura"
        ]

        return any(
            word in detected_text
            for word in wake_words
        )

    except Exception as error:

        print("Wake detection error:", error)

        return False


# ==================================================
# OLLAMA AI RESPONSE
# ==================================================

def get_ai_reply(user_text):

    try:

        with conversation_lock:

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            messages.extend(conversation_history[-10:])

            messages.append({
                "role": "user",
                "content": user_text
            })

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": AI_MODEL,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            reply = result["message"]["content"].strip()

            conversation_history.append({
                "role": "user",
                "content": user_text
            })

            conversation_history.append({
                "role": "assistant",
                "content": reply
            })

            return reply

    except requests.exceptions.ConnectionError:

        return "Ollama is not running. Please start Ollama."

    except Exception as error:

        print("AI error:", error)

        return "Sorry, something went wrong."


# ==================================================
# TEXT TO SPEECH
# ==================================================

def speak(text):

    try:

        with tts_lock:

            engine = pyttsx3.init("sapi5")

            engine.setProperty("rate", 165)

            engine.say(text)

            engine.runAndWait()

            engine.stop()

    except Exception as error:

        print("Speech error:", error)


# ==================================================
# AURA APPLICATION
# ==================================================

class AuraApp:

    def __init__(self, root):

        self.root = root

        self.root.title("AURA - AI Assistant")

        self.root.geometry("750x650")

        self.root.configure(bg="#171923")

        self.ui_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.busy_event = threading.Event()

        self.create_interface()

        self.root.after(
            100,
            self.process_ui_queue
        )

        # Start wake word listener

        self.wake_thread = threading.Thread(
            target=self.wake_listener,
            daemon=True
        )

        self.wake_thread.start()


    # ==================================================
    # CREATE GUI
    # ==================================================

    def create_interface(self):

        tk.Label(
            self.root,
            text="AURA",
            font=("Arial", 30, "bold"),
            fg="#7DD3FC",
            bg="#171923"
        ).pack(pady=12)

        tk.Label(
            self.root,
            text="Your English AI Assistant",
            font=("Arial", 11),
            fg="white",
            bg="#171923"
        ).pack()

        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg="#252A38",
            fg="white",
            insertbackground="white",
            state=tk.DISABLED
        )

        self.chat_area.pack(
            padx=15,
            pady=15,
            fill=tk.BOTH,
            expand=True
        )

        self.status_label = tk.Label(
            self.root,
            text="Starting Aura...",
            font=("Arial", 10),
            fg="#7DD3FC",
            bg="#171923"
        )

        self.status_label.pack(pady=5)

        self.message_entry = tk.Entry(
            self.root,
            font=("Arial", 13),
            bg="#252A38",
            fg="white",
            insertbackground="white"
        )

        self.message_entry.pack(
            padx=15,
            pady=8,
            fill=tk.X
        )

        self.message_entry.bind(
            "<Return>",
            lambda event: self.send_text()
        )

        button_frame = tk.Frame(
            self.root,
            bg="#171923"
        )

        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Send",
            command=self.send_text,
            width=12,
            bg="#0284C7",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Speak",
            command=self.start_manual_recording,
            width=12,
            bg="#059669",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_chat,
            width=12,
            bg="#475569",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app
        )


    # ==================================================
    # UI QUEUE
    # ==================================================

    def process_ui_queue(self):

        try:

            while True:

                action, data = self.ui_queue.get_nowait()

                if action == "message":

                    self.add_message(
                        data[0],
                        data[1]
                    )

                elif action == "status":

                    self.status_label.config(
                        text=data
                    )

                elif action == "clear":

                    self.chat_area.config(
                        state=tk.NORMAL
                    )

                    self.chat_area.delete(
                        "1.0",
                        tk.END
                    )

                    self.chat_area.config(
                        state=tk.DISABLED
                    )

        except queue.Empty:

            pass

        if not self.stop_event.is_set():

            self.root.after(
                100,
                self.process_ui_queue
            )


    # ==================================================
    # DISPLAY CHAT MESSAGE
    # ==================================================

    def add_message(self, sender, message):

        self.chat_area.config(
            state=tk.NORMAL
        )

        self.chat_area.insert(
            tk.END,
            sender + ": " + message + "\n\n"
        )

        self.chat_area.see(tk.END)

        self.chat_area.config(
            state=tk.DISABLED
        )


    # ==================================================
    # BUSY CONTROL
    # ==================================================

    def claim_busy(self):

        with state_lock:

            if self.busy_event.is_set():
                return False

            self.busy_event.set()

            return True


    def release_busy(self):

        with state_lock:

            self.busy_event.clear()

        self.ui_queue.put((
            "status",
            "Listening for Hey Aura..."
        ))


    # ==================================================
    # CLEAR CHAT
    # ==================================================

    def clear_chat(self):

        self.ui_queue.put((
            "clear",
            None
        ))

        with conversation_lock:

            conversation_history.clear()


    # ==================================================
    # SEND TEXT MESSAGE
    # ==================================================

    def send_text(self):

        text = self.message_entry.get().strip()

        if not text:
            return

        if not self.claim_busy():
            return

        self.message_entry.delete(
            0,
            tk.END
        )

        self.start_reply(text)


    # ==================================================
    # START REPLY
    # ==================================================

    def start_reply(self, text):

        self.ui_queue.put((
            "message",
            ("You", text)
        ))

        threading.Thread(
            target=self.reply_worker,
            args=(text,),
            daemon=True
        ).start()


    # ==================================================
    # AI REPLY WORKER
    # ==================================================

    def reply_worker(self, text):

        try:

            self.ui_queue.put((
                "status",
                "Aura is thinking..."
            ))

            reply = get_ai_reply(text)

            self.ui_queue.put((
                "message",
                ("Aura", reply)
            ))

            self.ui_queue.put((
                "status",
                "Aura is speaking..."
            ))

            speak(reply)

        except Exception as error:

            print("Reply error:", error)

        finally:

            self.release_busy()


    # ==================================================
    # MANUAL VOICE RECORDING
    # ==================================================

    def start_manual_recording(self):

        if not self.claim_busy():
            return

        threading.Thread(
            target=self.record_worker,
            daemon=True
        ).start()


    def record_worker(self):

        try:

            self.ui_queue.put((
                "status",
                "Listening... Speak now."
            ))

            with mic_lock:

                audio = sd.rec(
                    int(RECORD_SECONDS * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32"
                )

                sd.wait()

            text = transcribe_audio(audio)

            print("Recognized:", text)

            if not text:

                self.ui_queue.put((
                    "message",
                    ("Aura", "I could not hear you. Please try again.")
                ))

                return

            self.start_reply(text)

        except Exception as error:

            print("Recording error:", error)

            self.ui_queue.put((
                "status",
                "Microphone error. Please try again."
            ))

        finally:

            self.release_busy()


    # ==================================================
    # WAKE WORD LISTENER
    # ==================================================

    def wake_listener(self):

        self.ui_queue.put((
            "status",
            "Listening for Hey Aura..."
        ))

        while not self.stop_event.is_set():

            if self.busy_event.is_set():

                time.sleep(0.3)

                continue

            try:

                with mic_lock:

                    if self.stop_event.is_set():
                        break

                    audio = sd.rec(
                        int(WAKE_SECONDS * SAMPLE_RATE),
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32"
                    )

                    sd.wait()

                if self.stop_event.is_set():
                    break

                if self.busy_event.is_set():
                    continue

                if detect_wake_word(audio):

                    if not self.claim_busy():
                        continue

                    self.ui_queue.put((
                        "status",
                        "Hey Aura detected! Speak now."
                    ))

                    try:

                        with mic_lock:

                            command_audio = sd.rec(
                                int(RECORD_SECONDS * SAMPLE_RATE),
                                samplerate=SAMPLE_RATE,
                                channels=1,
                                dtype="float32"
                            )

                            sd.wait()

                        text = transcribe_audio(
                            command_audio
                        )

                        print("Recognized:", text)

                        if text:

                            self.start_reply(text)

                        else:

                            self.ui_queue.put((
                                "message",
                                (
                                    "Aura",
                                    "I could not hear you. Please try again."
                                )
                            ))

                    finally:

                        self.release_busy()

            except Exception as error:

                print("Wake listener error:", error)

                self.ui_queue.put((
                    "status",
                    "Microphone error. Retrying..."
                ))

                time.sleep(2)


    # ==================================================
    # CLOSE APP
    # ==================================================

    def close_app(self):

        self.stop_event.set()

        self.root.destroy()


# ==================================================
# START AURA
# ==================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = AuraApp(root)

    root.mainloop()
