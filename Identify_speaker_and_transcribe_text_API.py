# -*- coding: utf-8 -*-

# LOADING THE MODELS

from flask import Flask, request, jsonify, send_file
import torch
import whisper
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pydub import AudioSegment
from io import BytesIO
import os
import noisereduce as nr
import numpy as np
from rapidfuzz import fuzz
from scipy.io import wavfile
from scipy.signal import resample
import json
from datetime import datetime

app = Flask(__name__)

def logg_error(error):
    logs_dir = "logs"

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_name = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file_path = os.path.join(logs_dir, log_file_name)

    with open(log_file_path, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} - ERROR - {error}\n")



pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="PATH TO AUTH_TOKEN",                                                            # You´ll need an authority_token to use this modell.
    cache_dir=None
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline.to(device)

whisper_model = whisper.load_model("large")                                                         # Large for optimal transcription

def get_dynamic_threshold(w_segments, base_threshold=0.4, multiplier=0.1):
    durations = [segment['end'] - segment['start'] for segment in w_segments]
    median_duration = np.median(durations)
    return base_threshold + multiplier * median_duration

def match_segments(w_segments, diarization_segments, base_threshold=0.4):
    try:
        short_segment_threshold = get_dynamic_threshold(w_segments, base_threshold)
        speaker_texts = []
        speaker_mapping = {"Talare 1": None, "Talare 2": None}
        previous_speaker = None
        unknown_speaker = "Talare Okänd"

        for i, segment in enumerate(w_segments):
            whisper_start, whisper_end = segment['start'], segment['end']
            whisper_duration = whisper_end - whisper_start
            current_speaker = None

            if whisper_duration <= short_segment_threshold:
                current_speaker = previous_speaker or unknown_speaker
            else:
                max_overlap = 0
                for turn, _, speaker in diarization_segments.itertracks(yield_label=True):
                    overlap = max(0, min(whisper_end, turn.end) - max(whisper_start, turn.start))
                    if overlap / whisper_duration > max_overlap:
                        current_speaker, max_overlap = speaker, overlap / whisper_duration

                current_speaker = current_speaker or previous_speaker or unknown_speaker
                if speaker_mapping["Talare 1"] is None:
                    speaker_mapping["Talare 1"] = current_speaker
                elif speaker_mapping["Talare 2"] is None and current_speaker != speaker_mapping["Talare 1"]:
                    speaker_mapping["Talare 2"] = current_speaker

            speaker_role = (
                "Talare 1" if current_speaker == speaker_mapping["Talare 1"]
                else "Talare 2" if current_speaker == speaker_mapping["Talare 2"]
                else unknown_speaker
            )

            if i > 0 and (whisper_start - w_segments[i - 1]['end']) > 1.5:
                previous_speaker = None
            speaker_texts.append(f"{speaker_role}: {segment['text']}")
            previous_speaker = current_speaker

        return speaker_texts

    except Exception as e:
      logg_error(f"Error in match_segments: {e}")
      raise RuntimeError(f"Error occured while matching segments")

def allowed_file_and_convert_to_wav(file_path):
    allowed_extensions_dict = {
        "mp3": AudioSegment.from_mp3,
        "mp4": lambda f: AudioSegment.from_file(f, format="mp4"),
        "wma": lambda f: AudioSegment.from_file(f, format="wma"),
        "aac": lambda f: AudioSegment.from_file(f, format="aac"),
        "flac": lambda f: AudioSegment.from_file(f, format="flac"),
        "m4a": lambda f: AudioSegment.from_file(f, format="m4a"),
        "wav": None
    }

    if not os.path.isfile(file_path):
      logg_error(f"File at {file_path} was not found")
      raise FileNotFoundError(f"The file was not found.")

    filename = os.path.basename(file_path)
    file_extension = filename.rsplit(".", 1)[1].lower()

    if file_extension not in allowed_extensions_dict:
      logg_error(f"allowed_file_and_convert_to_wav(): Invalid file format")
      raise ValueError("Invalid file format. Valid formats: wav, wma, mp3, mp4, m4a, flac, aac")

    try:
        if file_extension == "wav":
            wav_file = BytesIO()
            with open(file_path, "rb") as f:
                wav_file.write(f.read())
            wav_file.seek(0)
            return wav_file
        else:
            audio = allowed_extensions_dict[file_extension](file_path)
            wav_file = BytesIO()
            audio.export(wav_file, format="wav")
            wav_file.seek(0)
            return wav_file
    except Exception as e:
      logg_error(f"allowed_file_and_convert_to_wav(): Error converting {file_path} to WAV format: {e}")
      raise RuntimeError(f"Error converting {file_path} to WAV format")

def preprocess_audio_for_pyannote(input_path, target_sample_rate=16000):
    try:
        sample_rate, audio_data = wavfile.read(input_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        processed_audio = resample(audio_data, int(len(audio_data) * target_sample_rate / sample_rate))
        return processed_audio, target_sample_rate
    except Exception as e:
      logg_error(f"preprocess_audio_for_pyannote(): Error processing audio for PyAnnote: {e}")
      raise RuntimeError(f"Error processing audio for PyAnnote")

def preprocess_audio_for_whisper(input_path, target_sample_rate=16000):
    try:
        audio_data, sample_rate = librosa.load(input_path, sr=target_sample_rate, mono=True)
        audio_data = librosa.effects.preemphasis(audio_data)
        if np.max(np.abs(audio_data)) == 0:
            raise ValueError("Audio data is silent or corrupted.")
        return (audio_data / np.max(np.abs(audio_data))) * 0.5, target_sample_rate
    except Exception as e:
      logg_error(f"preprocess_audio_for_whisper(): Error processing audio for Whisper: {e}")
      raise RuntimeError(f"Error processing audio for Whisper")

@app.route("/transcribe", methods=["POST"])
def transcribe():
  file_path = request.form.get("file_path")

  if not file_path:
    logg_error("No file path found")
    return "Invalid or missing file path", 400
  if not os.path.exists(file_path):
    logg_error("No file found in file path")
    return "File not found", 400

  wav_file_path = allowed_file_and_convert_to_wav(file_path)
  dia_processed, diarization_sr = preprocess_audio_for_pyannote(wav_file_path)
  whisper_processed_audio, whisper_sr = preprocess_audio_for_whisper(wav_file_path)

  diarization_audio_tensor = torch.tensor(dia_processed, dtype=torch.float32).unsqueeze(0)
  diarization_processed_audio = {"waveform": diarization_audio_tensor, "sample_rate": diarization_sr}

  diarization_segments = pipeline(diarization_processed_audio, num_speakers=2)
  result = whisper_model.transcribe(whisper_processed_audio, word_timestamps=True)
  whisper_segments = result["segments"]

  speaker_texts = match_segments(whisper_segments, diarization_segments, base_threshold=0.4)

  def remove_duplicates(speaker_texts):
      postprocessed_texts = []
      previous_text = []
      i = 0
      for i, text in enumerate(speaker_texts):
          if len(postprocessed_texts) == 0 or text[11:] != postprocessed_texts[-1][11:]:
              postprocessed_texts.append(text)

      return postprocessed_texts

  cleaned_text = remove_duplicates(speaker_texts)

# IDENTIFIYNG SPEAKERS AS "CUSTOMER" AND "AGENT"

  def identify_roles_by_keywords(cleaned_speaker_texts):
      keyword_scores = {
          "Talare 1": {"customer": 0, "agent": 0},
          "Talare 2": {"customer": 0, "agent": 0}
     }


      with open("PATH TO KEYWORDS_DATA", "r", encoding="utf-8") as file:                                          # PATH TO JSNOFILE KEYWORDS_DATA
          data = json.load(file)
      agent_words = [entry["original"].lower() for entry in data["agent_words"]]
      for entry in data["agent_words"]:
          agent_words.extend([variant.lower() for variant in entry["variations"]])
      customer_words = [entry["original"].lower() for entry in data["customer_words"]]
      for entry in data["customer_words"]:
          customer_words.extend([variant.lower() for variant in entry["variations"]])

      def fuzzy_match(sentence, word_list, threshold=95):
          score = 0
          for word in word_list:
              match_score = fuzz.partial_ratio(sentence, word)
              if match_score >= threshold:
                  score += match_score / 100
          return score

      for text in cleaned_speaker_texts:
          try:
              speaker, sentence = text.split(":", 1)
          except ValueError:
            logg_error(f"fuzzy_match(): Error while splitting text(speaker, sentence = text.split(':', 1))")
            continue

          sentence = sentence.lower().strip()
          speaker = speaker.strip()

          if speaker in keyword_scores:
              agent_score = fuzzy_match(sentence, agent_words)
              keyword_scores[speaker]["agent"] += agent_score

              customer_score = fuzzy_match(sentence, customer_words)
              keyword_scores[speaker]["customer"] += customer_score

      talare1_total = keyword_scores["Talare 1"]["agent"] + keyword_scores["Talare 1"]["customer"]
      talare2_total = keyword_scores["Talare 2"]["agent"] + keyword_scores["Talare 2"]["customer"]

      if talare1_total > talare2_total:
          if keyword_scores["Talare 1"]["agent"] > keyword_scores["Talare 1"]["customer"]:
              role1 = "Agent"
              role2 = "Customer"
          else:
              role1 = "Customer"
              role2 = "Agent"
      elif talare2_total > talare1_total:
          if keyword_scores["Talare 2"]["agent"] > keyword_scores["Talare 2"]["customer"]:
              role1 = "Customer"
              role2 = "Agent"
          else:
              role1 = "Agent"
              role2 = "Customer"
      else:
          if keyword_scores["Talare 1"]["agent"] > keyword_scores["Talare 2"]["seller"]:
              role1 = "Agent"
              role2 = "Customer"
          else:
              role1 = "Customer"
              role2 = "Agent"

      return {"Talare 1": role1, "Talare 2": role2}

  identified_roles = identify_roles_by_keywords(cleaned_text)

# TURNING SPEAKERS TO NEW NAMES

  def update_speaker_names(speaker_texts, identified_roles):
    updated_texts = []
    for text in speaker_texts:
      speaker, sentence = text.split(":", 1)
      new_speaker_name = identified_roles.get(speaker, speaker)
      updated_text = f"{new_speaker_name}: {sentence}"
      updated_texts.append(updated_text)
    return updated_texts

  clean_text_identified_roles = update_speaker_names(cleaned_text, identified_roles)

# Converting result to JSON

  dialogue_data = {"dialogue": []}

  for line in clean_text_identified_roles:
      if line.startswith("Agent"):
          person = "agent"
          text = line.split(":", 1)[1].strip()
      elif line.startswith("Customer"):
          person = "customer"
          text = line.split(":", 1)[1].strip()
      else:
          continue

      dialogue_data["dialogue"].append({
          "person": person,
          "text": text
      })

  with open("dialogue.json", "w", encoding="utf-8") as json_file:
      json.dump(dialogue_data, json_file, ensure_ascii=False, indent=4)

  return send_file("dialogue.json", as_attachment=True)

if __name__ == '__main__':
    app.run()