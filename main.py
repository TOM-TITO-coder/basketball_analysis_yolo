from utils import read_video, save_video
from trackers import PlayerTracker
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    
    # Define paths
    model_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/models/player_detector_model.pt'
    video_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/input_videos/video_1.mp4'
    output_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/output_videos/video_1.mp4'
    stubs_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/stubs/player_track_stubs.pkl'
    
    # Read Video
    video_frame, fps = read_video(video_path)

    # Initialize Tracker
    player_tracker = PlayerTracker(model_path)
    
    # Run Tracker
    player_tracks = player_tracker.get_object_tracks(video_frame,
                                                     read_from_stubs=True,
                                                     stubs_path=stubs_path
                                                     )
    print(player_tracks)  

    # Save Video
    save_video(video_frame, fps, output_path)

if __name__ == "__main__":
    main()