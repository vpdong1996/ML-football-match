from utils import read_video, save_video
from tracker import Tracker


def main():
    video_frames = read_video("input_videos/fb_match.mp4")
    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    output_annotated_frames = tracker.draw_annotations(video_frames, tracks)
    
    save_video(output_annotated_frames, "output_video/output.avi")


if __name__ == "__main__":
    main()
