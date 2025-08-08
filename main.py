from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer
)
from team_assigner import TeamAssigner

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    
    # Define paths
    player_model_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/models/player_detector_model.pt'
    ball_model_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/models/ball_detector_model.pt'
    video_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/input_videos/video_1.mp4'
    output_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/output_videos/output_v1.mp4'
    player_stubs_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/stubs/player_track_stubs.pkl'
    ball_stubs_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/stubs/ball_track_stubs.pkl'
    team_assignment_stubs_path = 'D:/ShelfStudy/AI Engineer/Computer_Vision/basketball_analysis/basketball_analysis/stubs/team_assignment_stubs.pkl'

    # Read Video
    video_frame, fps = read_video(video_path)

    # Initialize Tracker
    player_tracker = PlayerTracker(player_model_path)
    ball_tracker = BallTracker(ball_model_path)

    # Run Tracker
    player_tracks = player_tracker.get_object_tracks(video_frame,
                                                     read_from_stubs=True,
                                                     stubs_path=player_stubs_path
                                                     )
    ball_tracks = ball_tracker.get_object_tracks(video_frame,
                                                 read_from_stubs=True,
                                                 stubs_path=ball_stubs_path
                                                 )
    # Remove wrong ball detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames=video_frame,
        player_trackers=player_tracks,
        read_from_stub=True,
        stub_path=team_assignment_stubs_path
    )
    
    # Draw Output
    # Initialize Drawer
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    
    # Draw Player Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames=video_frame,
        tracks=player_tracks,
        player_assignment=player_assignment
    )
    # Draw Ball Tracks
    output_video_frames = ball_tracks_drawer.draw(
        video_frames=output_video_frames,
        tracks=ball_tracks
    )

    # Save Video
    save_video(output_video_frames, fps, output_path)

if __name__ == "__main__":
    main()