class PID:
    def __init__(self, p, i, d, integral_limit=100.0, output_limit=None):
        self.kp = p
        self.ki = i
        self.kd = d
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = integral_limit
        self.output_limit = output_limit

    def calc(self, current, target):
        error = target - current

        p_term = self.kp * error
        self.integral += error
        # 积分限幅
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        output = p_term + i_term + d_term
        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))
        return output