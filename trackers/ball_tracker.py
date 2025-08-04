from ultralytics import YOLO, RTDETR
import supervision as sv
import sys
sys.path.append("../")
from utils import read_stubs, save_stubs
class BallTracker:
    def __init__(self, model_path):
        self.model = RTDETR(model_path)
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(
                batch_frames, 
                conf=0.5
            )
            detections.extend(batch_detections) # extend() to add list to list, append() would add as a single element
        return detections
    
    def get_object_tracks(self, frames, read_from_stubs=False, stubs_path=None):
        # Read from stubs if available
        tracks = read_stubs(read_from_stubs, stubs_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
            
        # detect ball in frames
        detections = self.detect_frames(frames)
        tracks = []
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv['ball']:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
                
                if chosen_bbox is not None:
                    tracks[frame_num][1] = {"bbox": chosen_bbox} # Assuming track_id is always 1 for the ball
        
        save_stubs(stubs_path, tracks)
        
        return tracks
