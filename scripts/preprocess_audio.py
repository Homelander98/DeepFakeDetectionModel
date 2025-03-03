import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import os
import pandas as pd

def extract_audio(video_path, output_audio_path):
    """Extract audio from a video and save it as a WAV file."""
    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return False

        # Load the video file
        video = VideoFileClip(video_path)
        if video.audio is None:
            print(f"Warning: No audio stream found in {video_path}. Skipping audio extraction.")
            return False  # No audio stream, skip extraction

        # Extract and save the audio
        video.audio.write_audiofile(output_audio_path, logger=None)  # Disable logging for cleaner output
        print(f"Extracted audio from {video_path} to {output_audio_path}.")
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def segment_audio(audio_path, segment_length=5):
    """Segment audio into smaller chunks of a specified length (in seconds)."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        segments = []
        for i in range(0, len(y), sr * segment_length):
            segment = y[i:i + sr * segment_length]
            segments.append(segment)
        return segments, sr
    except Exception as e:
        print(f"Error segmenting audio from {audio_path}: {e}")
        return None, None

def extract_audio_features(audio_segment, sr):
    """Extract MFCC features from an audio segment."""
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from audio segment: {e}")
        return None

if __name__ == "__main__":
    # Load metadata
    metadata_file = "data/dataset.csv"
    if not os.path.exists(metadata_file):
        sys.exit(f"Error: Metadata file {metadata_file} not found. Exiting.")

    metadata = pd.read_csv(metadata_file)

    # Create output folders
    audio_folder = "data/audio"
    features_folder = "data/audio_features"
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(features_folder, exist_ok=True)

    # Process each video
    for index, row in metadata.iterrows():
        video_path = row["video_path"]
        label = row["label"]

        # Skip if the video file doesn't exist
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found. Skipping.")
            continue

        # Extract audio (if available)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(audio_folder, f"{video_name}.wav")

        # Check if the video has an audio stream
        has_audio = extract_audio(video_path, audio_path)

        # If the video has no audio, skip audio processing
        if not has_audio:
            print(f"Warning: No audio stream found in {video_path}. Skipping audio processing.")
            continue

        # Segment audio
        segments, sr = segment_audio(audio_path)
        if not segments:
            print(f"Warning: Audio segmentation failed for {audio_path}. Skipping.")
            continue

        # Extract audio features
        features = []
        for segment in segments:
            mfccs = extract_audio_features(segment, sr)
            if mfccs is not None:
                features.append(mfccs)

        # Skip if no features were extracted
        if not features:
            print(f"Warning: No features extracted for {audio_path}. Skipping.")
            continue

        # Save features
        features_path = os.path.join(features_folder, f"{video_name}_features.npy")
        np.save(features_path, np.array(features))
        print(f"Saved audio features for {video_path} to {features_path}.")