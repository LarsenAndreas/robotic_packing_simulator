from math import isclose, sqrt

import numpy as np
from conveyor.tray import Tray
from conveyor.veggie import Veggie


class Arm:
    def __init__(self, pos_mount: tuple[float, float], reach_bb: tuple[float, float, float, float], speed: float, seed):
        """Constructs an arm.

        Args:
            pos_mount (tuple[float, float]): The mounting position of the arm. The bounding box will be created from this position.
            reach_bb (tuple[float, float, float, float]): The bounding box of the arm reach. Defines the distance the arm can move in each direction. Assumes the following structure: (north, east, south, west)
            speed (float): The speed of the arm.
        """

        self.tasks: list[dict] = []
        self.pos_mount = pos_mount
        self.reach_bb = reach_bb
        self.speed = speed

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __str__(self):
        return f"ARM | mount={self.pos_mount} | bb={self.reach_bb}"  # | speed={self.speed} | tasks={len(self.tasks)}"

    def _getPos(self, pos_start: tuple[float, float], pos_stop: tuple[float, float], t_start: float, t: float) -> tuple[float, float]:
        """Calculates the position of the arm given a constant speed along the defined path.

        Args:
            pos_start (tuple[float, float]): Intial x,y-coordinates.
            pos_stop (tuple[float, float]): Terminal x,y-coordinates.
            t_start (float): Movement starting time.
            t (float): Current time.

        Returns:
            tuple[float, float]: Current x,y-coordinate of the arm.
        """
        x0, y0 = pos_start
        x1, y1 = pos_stop
        t0 = t_start

        dt = t - t0
        dx = x1 - x0
        dy = y1 - y0
        dist = dt * self.speed
        norm = sqrt(dx**2 + dy**2)

        x = x0 + dist * dx / norm
        y = y0 + dist * dy / norm

        return (x, y)

    def getState(self, t: float) -> dict:
        """Gets the current state of the arm. The state carries information about the following:
        - `pos` (tuple): The current x and y coordinate of the arm.
        - `done_working` (float): Timestamp when arm is done working. Only assigned if the arm is working at time t.
        - `pos_start` (tuple): Starting x,y coordinate of the current move.
        - `pos_stop` (tuple): Ending x,y coordinate of the current move.
        - `target_veggie` (Veggie): Current targeted veggie.
        - `target_tray` (Tray): Current targeted tray.


        Args:
            t (float): Current time.

        Returns:
            dict: State information.
        """

        state = {
            "pos": self.pos_mount,
        }

        # Tasks are sorted based on starting time
        for task in self.tasks:  # If there are no tasks, we simply return the mounting position
            t0, t1 = task["time_allocated"]  # The combined movement time
            if t < t0:  # Before task
                break
            elif t0 <= t < t1:  # In task
                state["done_working"] = t1
                state["target_veggie"] = task["target_veggie"]
                state["target_tray"] = task["target_tray"]
                for move in task["movement"]:
                    # Are we in the middle of this sub-move
                    if move["time_start"] <= t <= move["time_stop"]:
                        x0, y0 = move["pos_start"]
                        x1, y1 = move["pos_stop"]
                        state["pos_start"] = move["pos_start"]
                        state["pos_stop"] = move["pos_stop"]
                        state["pos"] = self._getPos(  # Update the position of the arm
                            pos_start=(x0, y0),
                            pos_stop=(x1, y1),
                            t_start=move["time_start"],
                            t=t,
                        )
                        break
            else:  # Task is already completed, thus we update arm position with final position of movement path
                state["pos"] = task["movement"][-1]["pos_stop"]

        return state

    def scheduleTask(self, movement: list[dict], tray: Tray, veggie: Veggie):
        """Schedules a task on the arm. Expects the structure of 'movement' to be a list of dicts.
        Each dict must contain the keys `time_start`, `time_stop`, `pos_start`, `pos_stop`.

        Args:
            movement (list[dict]): Movement path.
            tray (Tray): Tray to target.
            veggie (Veggie): Veggie to target.

        Raises:
            Exception: Debugging.

        Returns:
            dict: The scheduled task.
        """
        task = {
            "movement": movement,
            "target_veggie": veggie,
            "target_tray": tray,
            "time_allocated": (movement[0]["time_start"], movement[-1]["time_stop"]),
        }

        if not self.canReach(veggie._getPos(movement[0]["time_stop"])):  # This mostly debugging
            pos_calc = veggie._getPos(movement[0]["time_stop"])
            pos_move = movement[0]["pos_stop"]
            if not isclose(pos_calc[0], pos_move[0], abs_tol=1e-12):  # Skip if floating point error
                raise Exception(f"Not in reach!\n{pos_calc[0]=}\n{pos_move[0]=}")

        veggie.t_picked = movement[0]["time_stop"]  # Make sure the veggie is picked
        tray.packVeggie(veggie=veggie, t=movement[1]["time_stop"])  # Make sure the veggie is packed in tray

        self.tasks.append(task)
        self.tasks = sorted(self.tasks, key=lambda x: x["time_allocated"][0])  # Tasks should be sorted

        return task

    def canReach(self, pos: tuple):
        """Checks if the position is within the reachable area.

        Args:
            pos (tuple): x,y coordinate

        Returns:
            bool
        """
        north, east, south, west = self.reach_bb

        in_x = self.pos_mount[0] - west <= pos[0] < self.pos_mount[0] + east
        in_y = self.pos_mount[1] - south <= pos[1] < self.pos_mount[1] + north

        return in_x and in_y

    def beforeReach(self, pos: tuple):
        """Checks if the position is to the left the reachable area.

        Args:
            pos (tuple): x,y coordinate.

        Returns:
            bool
        """
        north, east, south, west = self.reach_bb

        in_x = pos[0] < self.pos_mount[0] - west
        in_y = self.pos_mount[1] - south <= pos[1] < self.pos_mount[1] + north

        return in_x and in_y

    def timeMove(self, pos_start: tuple, pos_stop: tuple):
        """Calculates time it takes to move from a to b.

        Args:
            pos_start (tuple): Starting position.
            pos_stop (tuple): Ending position.

        Returns:
            float: Movement time.
        """
        x0, y0 = pos_start
        x1, y1 = pos_stop
        dist = sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)  # Main man Pythagoras

        return dist / self.speed
