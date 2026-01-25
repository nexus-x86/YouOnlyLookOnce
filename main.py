from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

# The training data struggles with identifying the court lines on clay courts.
testing = 2
readFromData = True

def main():
    # Read Video
    input_video_path = f"input_videos/input_video{testing}.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and ball
    player_tracker = PlayerTracker(model_path='yolov8x') 
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=readFromData,
                                                     stub_path=f"tracker_stubs/player_detections{testing}.pkl")
    
    
    ball_tracker = BallTracker(model_path='models/yolov5_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=readFromData,
                                                     stub_path=f"tracker_stubs/ball_detections{testing}.pkl")
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    


    # Draw output

    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court kepoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, f"output_videos/output_video{testing}.avi")

if __name__ == "__main__":
    main()