import librosa
import numpy as np
from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def segment_audio(audio_path, segment_length=5):
    """
    Segments audio into smaller chunks of a specified length (in seconds).
    """
    y, sr = librosa.load(audio_path, sr=None)
    segments = []
    for i in range(0, len(y), sr * segment_length):
        segment = y[i:i + sr * segment_length]
        segments.append(segment)
    return segments

def extract_audio_features(audio_segment, sr):
    """
    Extracts MFCC features from an audio segment.
    """
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    return mfccs