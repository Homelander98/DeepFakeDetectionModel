import cv2
from mtcnn import MTCNN

def split_video_to_frames(video_path, output_folder):
    """
    Splits a video into individual frames and saves them to the output folder.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", frame)
        frame_count += 1
    cap.release()

def detect_and_crop_faces(frame, detector):
    """
    Detects faces in a frame and returns cropped face images.
    """
    faces = detector.detect_faces(frame)
    cropped_faces = []
    for face in faces:
        x, y, width, height = face['box']
        cropped_face = frame[y:y+height, x:x+width]
        cropped_faces.append(cropped_face)
    return cropped_faces

def compute_optical_flow(prev_frame, next_frame):
    """
    Computes optical flow between two consecutive frames.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def save_processed_video(frames, output_path, fps=30):
    """
    Saves a list of frames as a video file.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()