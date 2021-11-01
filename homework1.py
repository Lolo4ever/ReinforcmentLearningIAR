import math
import random
# import gym
# from gym import spaces, logger
# from gym.utils import seeding
import numpy as np
import time
import math

"""
Description:
    A cleaning robot in a fully observable environment. The robot cleaner makes decisions at each time
    step. At each step the robot decides whether it should :
        (1) actively search for area to clean
        (2) remain stationary and wait
        (3) go back to home base to recharge its battery. 
Observation:
    Type: Discrete(4)
    Num     Observation               Min                     Max
    0       Robot Position           (0,0)                    (9,9)
    1       Battery                    0                       100
    2       Dirtyness[x,y]             0                        5  
Actions:
    Type: Discrete(2)
    Num   Action
    0     Turn left
    1     Turn right
    2     Move foward
    3     Wait
    Note: The floor is automatically cleaned after the robot goes throw it.
Reward:
    The Dirtyness level of the square the robot's at. 
    0.5 if it is at home ???
Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
Episode Termination:
    The robot runs out of battery.
    Episode length is greater than 20000.
Solved Requirements:
    NOT
"""

ENERGY_PER_MOVE = 5


class DeadBatterError(Exception):
    pass


class WallE:
    """
    Class that define the robot position and direction
    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = "up"
        self.battery = 100

    def get_coordinates(self):
        return self.x, self.y

    def consume_energy(self):
        self.battery -= ENERGY_PER_MOVE
        if self.battery <= 0:
            raise DeadBatterError

    def move_forward(self):
        self.consume_energy()
        if self.direction == "up":
            self.y += 1
        if self.direction == "down":
            self.y -= 1
        if self.direction == "right":
            self.x += 1
        if self.direction == "left":
            self.x -= 1

    def turn_left(self):
        self.consume_energy()
        if self.direction == 'left':
            return 'down'
        if self.direction == 'down':
            return 'right'
        if self.direction == 'right':
            return 'up'
        if self.direction == 'up':
            return 'left'

    def turn_right(self):
        self.consume_energy()
        if self.direction == 'left':
            return 'up'
        if self.direction == 'up':
            return 'right'
        if self.direction == 'right':
            return 'down'
        if self.direction == 'down':
            return 'left'

    def wait(self):
        self.consume_energy()


class Board:
    """
    define a board
    """

    def __init__(self, height, length):
        self.height = height
        self.length = length
        self.matrix = np.zeros((height, length))  # dirtiness board
        self.robot = None

    def addRobot(self, robot: WallE, x, y):
        self.matrix[x, y] = 1
        self.robot = robot
        self.robot.x = x
        self.robot.y = y

    def room_initialization(self):
        for i in range(self.height):
            for j in range(self.length):
                self.matrix[i, j] = random.randint(0, 5)

        # add wall
        for index in range(4):
            self.matrix[index, 2] = np.inf

    def __str__(self):
        return str(self.matrix)

    def draw_map(self):
        map = ""
        for y in range(self.height):
            for x in range(self.length):
                if self.robot is not None and y == self.robot.y and x == self.robot.x:  # wall
                    map += " 🤖"
                elif self.matrix[y, x] == np.inf:  # wall
                    map += " 🧱"
                else:
                    map += "  "
            map += "\n"

        print(map)


def format_row(row):
    return '|' + '|'.join('{0:^5}'.format(x) for x in row) + '|'


def format_board(board):
    # for a single list with 9 elements uncomment the following line:
    # return '\n\n'.join(format_row(row) for row in zip(*[iter(board)]*3))
    # for a 3x3 list:
    return '\n\n'.join(format_row(row) for row in board)


def step():
    pass


def room_initialization(x, y):
    room = [[0 for i in range(x)] for j in range(y)]
    for i in range(x):
        for j in range(y):
            room[i][j] = random.randint(0, 5)
    room[0][2] = float('-inf')
    room[1][2] = float('-inf')
    room[2][2] = float('-inf')
    room[3][2] = float('-inf')
    room[HOMECOORD[0]][HOMECOORD[1]] = "X"
    return room


def move_forward(robot, direction, room):
    if direction == 'left' and robot[1] != 0 and room[robot[0]][robot[1] - 1] != float('-inf'):
        room[robot[0]][robot[1]] = 0
        room[robot[0]][robot[1] - 1] = "R"
        return (robot[0], robot[1] - 1)
    if direction == 'right' and robot[1] != 9 and room[robot[0]][robot[1] + 1] != float('-inf'):
        room[robot[0]][robot[1]] = 0
        room[robot[0]][robot[1] + 1] = "R"
        return (robot[0], robot[1] + 1)
    if direction == 'up' and robot[0] != 0 and room[robot[0] - 1][robot[1]] != float('-inf'):
        room[robot[0]][robot[1]] = 0
        room[robot[0] - 1][robot[1]] = "R"
        return (robot[0] - 1, robot[1])
    if direction == 'down' and robot[0] != 9 and room[robot[0] + 1][robot[1]] != float('-inf'):
        room[robot[0]][robot[1]] = 0
        room[robot[0] + 1][robot[1]] = "R"
        return (robot[0] + 1, robot[1])
    else:
        return robot


if __name__ == "__main__":
    # Environment Variables
    # Room Size
    X = 10
    Y = 10
    robot = WallE()
    room = Board(X, Y)
    room.room_initialization()

    print(room)
    room.addRobot(robot, 0, 0)
    room.draw_map()
    i = 0

    # while True:
    #     a = random.randint(0, 2)
    #     if a == 0:
    #         robot = move_forward(robot, direction, room)
    #     if a == 1:
    #         direction = turn_left(direction)
    #     if a == 2:
    #         direction = turn_right(direction)
    #
    #     print(format_board(room))
    #     print(robot)
    #     i += 1
    #     print(i)
    #     print("\n")
    #     time.sleep(0.2)

"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
"""
