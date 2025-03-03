from scripts.preprocess_video import split_video_to_frames
from scripts.preprocess_audio import extract_audio, segment_audio, extract_audio_features
from scripts.train import train
from scripts.evaluate import evaluate_model
from scripts.train_gan import train_gan
from scripts.evaluate_gan import evaluate_gan

if __name__ == "__main__":
    # Preprocess video and audio
    split_video_to_frames("data/raw_videos/sample.mp4", "data/processed_frames")
    extract_audio("data/raw_videos/sample.mp4", "data/audio/sample.wav")
    segments = segment_audio("data/audio/sample.wav")
    features = [extract_audio_features(segment, 22050) for segment in segments]

    # Train GAN
    train_gan()

    # Train and evaluate deepfake detection model
    train()
    evaluate_model()

    # Evaluate GAN
    evaluate_gan()