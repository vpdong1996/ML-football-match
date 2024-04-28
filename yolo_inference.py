from ultralytics import YOLO

model = YOLO("models/best.pt")
result = model.predict("input_videos/fb_match.mp4", save=True)
