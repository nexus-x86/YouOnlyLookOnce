from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    # Read Video
    input_video_path = "input_videos/input_video2.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and ball
    player_tracker = PlayerTracker(model_path='yolov8x') 
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    
    
    ball_tracker = BallTracker(model_path='models/yolov5_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    
    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])


    # Draw output

    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court kepoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)


    save_video(output_video_frames, "output_videos/output_video2.avi")

if __name__ == "__main__":
    main()