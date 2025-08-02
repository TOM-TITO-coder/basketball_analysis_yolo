from ultralytics import YOLO, RTDETR
import supervision as sv
import sys
sys.path.append("../")
from utils import read_stubs, save_stubs

class PlayerTracker:
    def __init__(self, model_path):
        self.model = RTDETR(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]  # take a batch of frames, frames[0:20], frames[20:40], etc.
            batch_detections = self.model.predict(batch_frames, conf=0.5) # that mean we can process 20 frames at a time
            detections+= batch_detections
        return detections
    
    def get_object_tracks(self, frames, read_from_stubs=False, stubs_path=None):
        
        tracks = read_stubs(read_from_stubs, stubs_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
        
        detections = self.detect_frames(frames)
        tracks = []
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            detections_supervision = sv.Detections.from_ultralytics(detection) # convert detection from Ultralytics format to Supervision format
            
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision) # update tracker with detections
            
            tracks.append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3] 
                track_id = frame_detection[4] # 4 is the index for track id in the detection tuple, e.g., ((x1, y1), (x2, y2), conf, cls_id, track_id)
                
                if cls_id == cls_names_inv['player']:
                    tracks[frame_num][track_id] = {"box": bbox}

        save_stubs(stubs_path = stubs_path, object = tracks)

        return tracks