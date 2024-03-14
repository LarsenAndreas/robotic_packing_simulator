from math import ceil, sqrt

import numpy as np
from conveyor.arm import Arm
from conveyor.tray import Tray
from conveyor.veggie import Veggie


class System:
    def __init__(self, length_buffer: float, width_belt: float, speed_belt: float, cutoff_probability: float, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.length_buffer = length_buffer
        self.width_belt = width_belt
        self.kind_veggies = ["cucumber", "tomato", "eggplant"]
        self.speed_belt = speed_belt
        self.cutoff_probability = cutoff_probability

        self.arms: list[Arm] = []
        self.trays: list[Tray] = []
        self.veggies: list[Veggie] = []
        self.length_belt: float = None
        self.belt_start: float = None
        self.belt_end: float = None

    def genArms(self, pos_mount: list[tuple], reach_bb: list[tuple], speed: list[float]):
        """Generates and assigns robotic arms to the system. Note that the length of each input list must be identical. This also sets the length of the belt to the total horizontal reach of the arms.

        Args:
            pos_mount (list[tuple]): [(x,y), ...]. The mounting x,y coordinates of the arms. The bounding box will be created from this position.
            reach_bb (list[tuple]): [(north, east, south, west), ...]. The bounding box of each arms reach. Defines the distance the arm can move in each direction. .
            speed (list[float]): The speed of the arms.
        """

        if not len(pos_mount) == len(reach_bb) == len(speed):
            raise Exception("Iteratible lengths does not match!")

        for pos, bb, s in zip(pos_mount, reach_bb, speed):
            arm = Arm(pos_mount=pos, reach_bb=bb, speed=s, seed=self.rng.integers(0, int(2**32)))  # Seed is tied to system RNG for reproducability
            self.arms.append(arm)

        arm_left = min(self.arms, key=lambda x: x.pos_mount[0])  # Leftmost arm
        arm_right = max(self.arms, key=lambda x: x.pos_mount[0])  # Rightmost arm
        self.belt_start = arm_left.pos_mount[0] - arm_left.reach_bb[3]
        self.belt_end = arm_right.pos_mount[0] + arm_right.reach_bb[1]
        self.length_belt = abs(self.belt_start - self.belt_end)  # Belt length is forced to be exactly arm coverage

    def genTrays(self, count: int, spacing: float, max_veggies: int):
        """Generates and assigns trays to the system.

        Args:
            count (int): Total number of trays.
            spacing (float): Spacing between each tray. This will increase how many trays are availible to each arm at any given time.
            max_veggies (int): Maximum number of veggies which can be assigned. Must be >2.
        """

        if not self.length_belt:
            raise Exception("The belt has not been assigned a length yet! Either manually assign a length, or run the genArms method.")

        x = self.belt_end
        y = self.width_belt / 2 + 5  # How far from the main belt each tray will be located
        i = 1
        loc = 1
        while i <= count:
            spec_content = self.rng.choice(self.kind_veggies, size=self.rng.integers(2, max_veggies, endpoint=True))
            spec_weight = sum([Veggie.getProperties(v, rng=self.rng)["weight"] for v in spec_content])
            tray = Tray(
                pos_init=(x - (i + i % 2) * spacing, loc * y), dist_move=spacing, spec_content=spec_content, spec_weight=spec_weight, seed=self.rng.integers(0, int(2**32))
            )  # Seed is tied to system RNG for reproducability
            self.trays.append(tray)
            i += 1
            loc *= -1

    def genVeggies(self, count: int, t_min: float, t_max: float):
        """Generates and assigns veggies to the system. The veggies will reach the main belt uniformly from 't_min' to 't_max'.

        Args:
            count (int): Number of veggies spawned.
            t_min (float): First time a veggie can spawn.
            t_max (float): Last time a veggie can spawn.
        """

        times = self.rng.uniform(t_min, t_max, size=count)
        locations = self.rng.uniform(-self.width_belt / 2, self.width_belt / 2, size=count)
        kinds = self.rng.choice(self.kind_veggies, size=count, replace=True)
        for t, y, v in zip(times, locations, kinds):
            veggie = Veggie(t_spawn=t, pos_init=(0, y), kind=v, speed=self.speed_belt, seed=self.rng.integers(0, int(2**32)))  # Seed is tied to system RNG for reproducability
            self.veggies.append(veggie)

    def getSystemState(self, t: float, **kwargs) -> dict:
        """Gets the current state of the system. This only returns the veggies and trays which currently lies on/beside either the buffer of the belt. The system state includes the following keys:
        - `t` (float): The time the system state corresponds to.
        - `arms` (list[dict,]): The arms associated with the system.
        - `veggies` (list[dict, ]): The vegetables currently visible to the system.
        - `trays` (list[dict, ]): The trays currently visible to the system.

        Note that for each key, a list of states corresponding to each arm/veggie/tray at t is generated. See arm/veggie/tray.getState() for more info.

        Args:
            t (float): Current time.
            **kwargs: Passed to tray.getState().

        Returns:
            dict: System state.
        """
        state_system = {
            "t": t,
            "arms": {},
            "veggies": {},
            "trays": {},
        }

        for arm in self.arms:  # Every arm is always included
            state = arm.getState(t)
            state_system["arms"][arm] = state

        for tray in self.trays:
            state = tray.getState(t, **kwargs)

            # Trays must be beside the buffer/belt
            if state["pos"] and self.belt_start - self.length_buffer <= state["pos"][0] < self.belt_end:
                state_system["trays"][tray] = state
            # if state["pos"] and self.belt_start <= state["pos"][0] < self.belt_end:
            #     state_system["trays"][tray] = state

        for veggie in self.veggies:
            state = veggie.getState(t)
            # Veggie must be on the buffer/belt, and not be picked
            if (not state["t_picked"] or t < state["t_picked"]) and (self.belt_start - self.length_buffer <= state["pos"][0] < self.length_belt):
                state_system["veggies"][veggie] = state

        return state_system

    def _calcTimeInterceptVeggie(self, pos_veggie: tuple, pos_arm: tuple, speed_veggie: float, speed_arm: float) -> float | None:
        """Calculates the time to intercept a vegetable from an arm.
        We assume that the vegetable is moving at a constant pace from left to right, and calculates how the arm should move to intercept the veggie as fast as possible.
        This is then expressed as a time-delta.

        Args:
            pos_veggie (tuple): Position of the veggie.
            pos_arm (tuple): x,y coordinate of the arm.
            speed_veggie (float): Speed of the conveyor belt.
            speed_arm (float): Speed of the arm.

        Returns:
            float: Time to intercept.
        """

        d = self.speed_belt**2 * (2 * pos_arm[0] - 2 * pos_veggie[0]) ** 2 - 4 * (speed_arm**2 - speed_veggie**2) * (
            -pos_veggie[0] ** 2 + 2 * pos_arm[0] * pos_veggie[0] - pos_arm[0] ** 2 - pos_veggie[1] ** 2 - pos_arm[1] ** 2 + 2 * pos_veggie[1] * pos_arm[1]
        )  # Discriminant

        if d < 0:  # No real solutions
            return None

        sqrt_d = sqrt(d)
        b = 2 * speed_veggie * pos_veggie[0] - 2 * speed_veggie * pos_arm[0]
        a = speed_veggie**2 - speed_arm**2

        if a == 0:
            raise Exception("PLEASE HANDLE ME!")

        # We basically solve a 2nd degree polynomial
        dt_intercept_minus = (-b - sqrt_d) / (2 * a)
        dt_intercept_plus = (-b + sqrt_d) / (2 * a)

        dt_intercept = min(dt_intercept_minus, dt_intercept_plus)
        if dt_intercept < 0:
            dt_intercept = max(dt_intercept_minus, dt_intercept_plus)

        return dt_intercept

    def getTimeInterceptVeggie(self, t: float, veggie: Veggie, arm: Arm, system_state: dict = None) -> dict | None:
        """Get the information corresponding to when the arm can first intercept the veggie, given the state of the system at time t.
        The movement information contains the following:
        - `t_move_start` (float): Timestamp of when the arm should start moving.
        - `dt_move` (float): The time-delta it takes to move.
        - `t_move_end` (float): Timestamp of when the arm will intercept the veggie.
        - `pos_start` (tuple): Initial x,y coordinate of the arm.
        - `pos_end` (tuple): x,y coordinate of the interception point.

        Args:
            t (float): Current time.
            veggie (Veggie): Target veggie.
            arm (Arm): Arm to utilize.
            system_state (dict, optional): Allow passing the system state directly. This is mostly to speed up repeated calculations. Defaults to None.

        Returns:
            float | None: The movement information.
        """
        state_arm = arm.getState(t) if not system_state else system_state["arms"][arm]

        state_veggie = veggie.getState(t) if not system_state else system_state["veggies"][veggie]
        if state_veggie["t_picked"]:
            # Veggie is already picked
            return None

        dt_move = self._calcTimeInterceptVeggie(pos_veggie=state_veggie["pos"], pos_arm=state_arm["pos"], speed_veggie=veggie.speed, speed_arm=arm.speed)
        if not dt_move:
            # Veggie cannot be intercepted
            return None

        pos_veggie_intercept = veggie._getPos(t + dt_move)  # Position of the veggie given the time it takes to intercept
        if arm.canReach(pos_veggie_intercept):
            # Veggie is within reach
            t_move_start = t
            t_move_end = t + dt_move
            return {
                "t_move_start": t_move_start,
                "dt_move": dt_move,
                "t_move_end": t_move_end,
                "pos_start": state_arm["pos"],
                "pos_end": pos_veggie_intercept,
            }

        if arm.beforeReach(pos_veggie_intercept):
            # Veggie will be in reach later
            reach_left = arm.pos_mount[0] - arm.reach_bb[3]  # Leftmost position arm can reach
            pos_arm_intercept = (reach_left, pos_veggie_intercept[1])  # Intercept position
            dt_entry = (reach_left - state_veggie["pos"][0]) / veggie.speed  # Time until veggie enters reach
            state_arm_entry = arm.getState(t + dt_entry)  # State of the arm when veggie enters reach

            if state_arm_entry.get("done_working"):  # Arm is already working as veggie enters reach
                return self.getTimeInterceptVeggie(t=state_arm_entry["done_working"], veggie=veggie, arm=arm)  # Check the time after finished work
            else:  # Arm is not working as veggie enters reach
                dt_move = arm.timeMove(pos_start=state_arm_entry["pos"], pos_stop=pos_arm_intercept)  # Time to move arm from current pos to interception pos
                t_move_start = t + dt_entry - dt_move  # Move starts before the veggie is within reach. dt_entry > dt_move because of main man Pythagoras
                t_move_end = t + dt_entry  # Veggie is picked exactly as it enters reach
                return {
                    "t_move_start": t_move_start,
                    "dt_move": dt_move,
                    "t_move_end": t_move_end,
                    "pos_start": state_arm_entry["pos"],
                    "pos_end": pos_arm_intercept,
                }

        # Veggie is past arm reach
        return None

    def getTimeInterceptTray(self, t: float, tray: Tray, arm: Arm, pos_arm: tuple = None) -> tuple[float, float, float]:
        """Get the information corresponding to when the arm can intercept the tray, given the state of the system at time t.

        Args:
            t (float): Current time.
            tray (Tray): Target tray.
            arm (Arm): Arm to utilize.
            pos_arm (tuple, optional): Allows passing the arm position directly. Defaults to None.

        Returns:
            tuple[float, float, float]:
                - Timestamp of when the arm should start moving.
                - The time-delta it takes to move.
                - Timestamp of when the arm will intercept the tray.
        """

        pos_arm = arm._getPos(t) if not pos_arm else pos_arm

        state_tray = tray.getState(t)
        pos_tray = state_tray["pos"]
        moves_performed = state_tray["count_shifts"]

        dt_wait = 0

        if arm.beforeReach(pos_tray):
            # Tray is within reach
            reach_left = arm.pos_mount[0] - arm.reach_bb[3]  # Leftmost position arm can reach
            moves_required = ceil((reach_left - pos_tray[0]) / tray.dist_move)  # Absolute moves required for tray to reach leftmost pos
            moves_remaining = len(tray.t_moves) - moves_performed  # Remaining moves for tray to reach leftmost pos
            if moves_remaining < moves_required:  # Tray is not scheduled to be within reach
                return None

            pos_tray_moves = [pos_tray[0] + moves_required * tray.dist_move, pos_tray[1]]  # Pos of the tray after absolute required moves
            time_tray_moves = tray.t_moves[moves_performed + moves_required - 1]  # Time at which tray is at leftmost pos

            pos_tray = pos_tray_moves
            dt_wait = time_tray_moves - t  # Update wait time for tray to reach pos
            # arm.canReach should now be True

        if arm.canReach(pos_tray):
            reach_right = arm.pos_mount[0] + arm.reach_bb[1]  # Rightmost position arm cam reach
            while pos_tray[0] < reach_right:  # While the tray is reachable
                dt_move = arm.timeMove(pos_start=pos_arm, pos_stop=pos_tray)  # Time to move to tray

                if dt_move < dt_wait:  # Wait is longer than move time
                    t_move_start = t + dt_wait - dt_move
                    t_move_end = t + dt_wait
                    return t_move_start, dt_move, t_move_end
                else:  # Move time is longer than wait, thus we need to check if the tray has moved
                    state_tray_move = tray.getState(t + dt_move)  # Position after arm movement
                    if pos_tray == state_tray_move["pos"]:  # If the tray does change position while the arm is moving
                        t_move_start = t
                        t_move_end = t + dt_move
                        return t_move_start, dt_move, t_move_end
                    else:  # Tray has moved while arm was moving
                        dt_wait = tray.t_moves[state_tray_move["count_shifts"] - 1] - t
                        pos_tray[0] += tray.dist_move

        return None

    def moveTrays(self, loc: str, t: float):
        """Schedule a move on all trays in a row.

        Args:
            loc (str): (bot, top). Position of the trays.
            t (float): Time to schedule move.
        """
        for tray in self.trays:
            if tray.loc == loc:
                tray.scheduleMove(t)

    def clearUnfinishedTasks(self, t: float):
        """Clears scheduled tasks from all arms, trays, and veggies, where the tasks are to be performed after the current time.

        Args:
            t (float): Current time.
        """

        for tray in self.trays:
            tray.t_moves = [t_move for t_move in tray.t_moves if t_move < t]
            tray.packed = [(v, t_pack) for (v, t_pack) in tray.packed if t_pack < t]

        for arm in self.arms:
            arm.tasks = [task for task in arm.tasks if task["time_allocated"][0] < t]

        for veggie in self.veggies:
            if veggie.t_picked is not None and veggie.t_picked >= t:
                veggie.t_picked = None
