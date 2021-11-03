import argparse
import logging
import random
import numpy as np
import time

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
    The room is completely cleen
Solved Requirements:
    NOT
"""

ENERGY_PER_MOVE = 5
ROOM_SIZE = (10, 10)
HOME = (1, 1)
MAX_ITERATION = 20_000

class DeadBatterError(Exception):
    pass


class WallE:
    """
    Class that define the robot position and direction
    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = "south"
        self.battery = 100
        self.charger = HOME

    def get_coordinates(self):
        return self.x, self.y

    def consume_energy(self):
        self.battery -= ENERGY_PER_MOVE
        if self.battery <= 0:
            raise DeadBatterError

    def move_forward(self):
        self.consume_energy()
        if self.direction == "north":
            self.y -= 1
        if self.direction == "south":
            self.y += 1
        if self.direction == "east":
            self.x += 1
        if self.direction == "west":
            self.x -= 1

    def turn_left(self):
        self.consume_energy()
        if self.direction == 'west':
            self.direction =  'south'
        if self.direction == 'south':
             self.direction = 'east'
        if self.direction == 'west':
            self.direction = 'north'
        if self.direction == 'north':
            self.direction = 'west'

    def turn_right(self):
        self.consume_energy()
        if self.direction == 'west':
            self.direction = 'north'
        if self.direction == 'north':
            self.direction = 'east'
        if self.direction == 'east':
            self.direction = 'south'
        if self.direction == 'south':
            self.direction = 'west'

    def wait(self):
        pass


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
        self.robot = robot
        self.robot.x = y
        self.robot.y = x

    def room_initialization(self):
        for i in range(self.height):
            for j in range(self.length):
                self.matrix[i, j] = random.randint(0, 5)

        # add wall
        for index in range(4):
            self.matrix[index, 2] = np.inf

    def move(self, action):
        old_state = self.robot.x, self.robot.y

        getattr(self.robot, action)()
        # check if robot out of room
        if not (0 <= self.robot.x < 10 and 0 <= self.robot.y < 10):  # out of room
            self.robot.x = old_state[0]
            self.robot.y = old_state[1]
            logging.debug(f"robot out of room ")
        if self.matrix[self.robot.y, self.robot.x] == np.inf:
            self.robot.x = old_state[0]
            self.robot.y = old_state[1]
            logging.debug(f"collision with  wall avoided...")
        cleaned = self.clean_tile(self.robot.x, self.robot.y)
        logging.debug(f"robot in {self.robot.get_coordinates()}, {self.robot.direction}")
        return cleaned
        
    def clean_tile(self, x , y):
        cleaned = self.matrix[y,x]
        self.matrix[y,x] = 0
        return cleaned

    def is_room_clean(self):
        for row in self.matrix:
            for tile in row:
                if tile != np.inf or tile != 0:
                    return False
        return True

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
        if self.robot is not None:
            print(f"robot direction: {self.robot.direction}")

    def start_simulation(self, engine):
        try:
            logging.info("starting simulation...")
            robot = WallE()
            logging.debug("initializing room...")
            self.room_initialization()
            robot.direction = "south"
            logging.debug("room initialized")
            self.addRobot(robot, 0, 0)
            logging.debug("Wall-E added to the map")
            logging.info("Finished initializing environment, starting loop...")
            self.draw_map()
            rewards = []

            iteration_count = 0
            while not self.is_room_clean():
                logging.debug(f"staring iteration {iteration_count}...")
                action = engine(self)
                logging.debug(f"engine decided: {action}, executing...")

                reward = room.move(action)
                rewards.append(reward)
                logging.debug(f"finished execution with reward: {reward}, total rewards: {sum(rewards)}")

                if iteration_count > MAX_ITERATION:
                    logging.info(f"max iteration reached")
                    break

                time.sleep(0.2)

        except DeadBatterError:
            logging.error("Battery dead ")
        
        finally:
            logging.info(f"simulation finished 🥳")



def random_engine(board: Board):
    return random.choice(["move_forward", "turn_right", "turn_left"])
    # return random.choice(["move_forward"])


def super_engine(board: Board):
    """ TODO
    :param board: Board object with robot that describe the current state of the board and the robots position
    :return action: one of the following commands: move_forward turn_right turn_left wait that going to be executed
    before next loop
    """
    pass

def generate_states():
    states = []
    for i in range(10):
        for j in range(10):
            for k in range(0, 100, 5):
                states.append((i,j,k))
    return states

def dynamic_prog(board: Board, robot: WallE):
    V_function = []
    states = generate_states()
    for i in range(100):
        for s in states:
            states_prime = 
            pass

def sum_dynamic_prog(s, states):
    for s_prime in states:
        
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s--[%(levelname)s]: %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    # Environment Variables
    room = Board(*ROOM_SIZE)
    room.start_simulation(random_engine)

