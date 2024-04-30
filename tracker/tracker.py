from utils import get_center_of_bbox, get_bbox_width
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import pandas as pd
import numpy as np

sys.path.append("../")


class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_pos):
        ball_pos = [x.get(1, {}).get('bbox', []) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()

        ball_pos = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos

    def detect_frames(self, frames):
        batchSize = 20
        detections = []

        for i in range(0, len(frames), batchSize):
            detections_batch = self.model.predict(
                frames[i:i+batchSize], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for index, detection in enumerate(detections):
            cls_names = detection.names
            cls_name_inversion = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to player object
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_id == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_name_inversion["player"]

            # Track objects - add tracker object to detection
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                boundingBox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inversion["player"]:
                    tracks["players"][index][track_id] = {"bbox": boundingBox}
                if cls_id == cls_name_inversion["referee"]:
                    tracks["referees"][index][track_id] = {"bbox": boundingBox}

            for frame_detection in detection_supervision:
                boundingBox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inversion['ball']:
                    tracks["ball"][index][1] = {"bbox": boundingBox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for index, frame in enumerate(video_frames):
            # The copy function here to make sure the object in *video_frames* will not be modified
            frame = frame.copy()

            player_dict = tracks["players"][index]
            referee_dict = tracks["referees"][index]
            ball_dict = tracks["ball"][index]

            # Draw player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], (0, 0, 255))

            # Draw referee
            for track_id, ref in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, ref["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12

            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # 3 vertices of triangle
        triangle_points = np.array([[x, y], [x-10, y-20], [x+10, y-20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
