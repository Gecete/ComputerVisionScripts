from google.cloud import speech_v1p1beta1 as speech
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C://Users/gecete/Desktop/creden.json"
client = speech.SpeechClient()

speech_file = 'C://Users/gecete/Desktop/audio2.wav'

with open(speech_file, 'rb') as audio_file:
    content = audio_file.read()

audio = speech.types.RecognitionAudio(content=content)

config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code='en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count=2)

print('Waiting for operation to complete...')
response = client.recognize(config, audio)

# The transcript within each result is separate and sequential per result.
# However, the words list within an alternative includes all the words
# from all the results thus far. Thus, to get all the words with speaker
# tags, you only have to take the words list from the last result:
result = response.results[4]

words_info = result.alternatives[0].words

# Printing out the output:
for word_info in words_info:
    print("word: '{}', speaker_tag: {}".format(word_info.word,
                                               word_info.speaker_tag))
