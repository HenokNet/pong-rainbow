import numpy as np

class StateEncoder:
    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        self.ball_speed_scale = 15.0
        self.future_steps = 20

    def get_state(self, ball, ai_paddle, ball_speed_x, ball_speed_y):
        predicted_y = ball.y + (ball_speed_y * self.future_steps)
        predicted_y = np.clip(predicted_y, 0, self.game_height)
        
        if ball_speed_x == 0:
            angle = 0.0
        else:
            angle = np.arctan(ball_speed_y / abs(ball_speed_x)) / (np.pi/2)

        return np.array([
            (ball.x - self.game_width/2) / (self.game_width/2),
            (ball.y - self.game_height/2) / (self.game_height/2),
            (ai_paddle.centery - self.game_height/2) / (self.game_height/2),
            ball_speed_x / self.ball_speed_scale,
            ball_speed_y / self.ball_speed_scale,
            (predicted_y - self.game_height/2) / (self.game_height/2),
            angle
        ], dtype=np.float32)