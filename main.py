from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def main():
    video_frames = read_video("input_videos/fb_match.mp4")
    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # Interpolate ball position
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_index, player in enumerate(tracks["players"]):
        for player_id, track in player.items():
            team = team_assigner.get_player_team(
                video_frames[frame_index], track["bbox"], player_id)

            tracks["players"][frame_index][player_id]['team'] = team
            tracks["players"][frame_index][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    for frame_index, player in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_index][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(
            player, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_index][assigned_player]["has_ball"] = True

    # Draw output
    output_annotated_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_annotated_frames, "output_video/output.avi")


if __name__ == "__main__":
    main()
