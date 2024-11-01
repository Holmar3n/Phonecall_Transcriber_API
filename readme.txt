### Introduction

The system uses two models: one for transcription and one for voice analysis. The API accepts a file path, and the audio files are preprocessed to work better with the models. 
The models create segments that are then matched together using the match_segments() function. With the help of the identify_roles_by_keywords() function, the roles of Speaker 1 and Speaker 2 are determined (e.g., agent or customer). 
The final list is then converted into a JSON file, which is sent back as the output.

### Installations/Models:
- `torch`
- `pyannote.audio`
- `rapidfuzz`
- `numpy`
- `pydub`
- `whisper` [https://github.com/openai/whisper.git](https://github.com/openai/whisper.git)
- `flask`
- `noisereduce`

### Authorization:
PyAnnote requires creating a token on Hugging Face and accepting their terms of service.

### Additional Notes:
Some audio files are included for testing. Some are high quality, while others are lower quality. The `identify_roles_by_keywords()` function requires the path to `keywords_data.json`.

*Note*: I've only been able to run the code on Google Colab's paid version due to the high performance requirements of the models.
