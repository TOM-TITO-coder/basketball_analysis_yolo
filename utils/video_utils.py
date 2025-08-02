import cv2 
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True: 
        ret, frame = cap.read() # ret is a boolean indicating if the frame was read correctly
        if not ret:             # if no frame is returned, break the loop
            break
        
        frames.append(frame) 
    
    fps = cap.get(cv2.CAP_PROP_FPS)  
    cap.release()
    return frames, fps

def save_video(output_frames, fps, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_frames[0].shape[1], output_frames[0].shape[0]))

    for frame in output_frames:
        out.write(frame)

    out.release()