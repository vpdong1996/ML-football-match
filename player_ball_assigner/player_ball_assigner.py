from utils import get_center_of_bbox, measure_distance
import sys
sys.path.append("../")


class PlayerBallAssigner():
    def __init__(self) -> None:
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)

        minium_distance = 9999
        assign_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_pos)
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_pos)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minium_distance:
                    minium_distance = distance
                    assign_player = player_id

        return assign_player
