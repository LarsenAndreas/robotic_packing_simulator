import numpy as np
from conveyor.veggie import Veggie


class Tray:
    def __init__(self, pos_init: tuple[float, float], dist_move: float, spec_content: list[str,], spec_weight: float, seed: int):
        """Constructs a tray.

        Args:
            pos_init (tuple[float, float]): The initial position of the tray.
            dist_move (float): How much the tray will move when issued a move command.
            spec_content (list[str,]): The content specification. See conveyor.Veggie.getProperties for options.
            spec_weight (float): The required weight.
            seed (int): For reproducablity.
        """

        self.pos_init = pos_init
        self.dist_move = dist_move
        self.spec_content = tuple(spec_content)
        self.spec_weight = spec_weight
        self.loc = "bot" if pos_init[1] < 0 else "top"  # Assumes trays are on either side of the conveyor belt
        self.discarded = False  # If the tray has been discarded. Controlled by the controller

        self.t_moves: list[float] = []  # [t0, t1, t2, ...]
        self.packed: list[tuple[Veggie, float]] = []  # [(veggie0, t0), (veggie1, t1), ...]

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __str__(self):
        return f"TRAY | spec_weight={self.spec_weight} | spec_pack={self.spec_content}"

    def _getPos(self, t: float) -> tuple:
        """Get the current position.

        Args:
            t (float): Current time.

        Returns:
            tuple: Current x,y-coordinate of the tray.
        """
        pos = list(self.pos_init)
        for t_move in self.t_moves:  # sorted by t
            if t < t_move:
                break
            pos[0] += self.dist_move

        return pos

    def getState(self, t: float, time_dependent_specs=False) -> dict:
        """Gets the current state of the tray. The state carries information about the following:
        - `pos` (tuple): The current x and y coordinate of the tray.
        - `remain_weight` (float): Remaining weight. If `time_dependent_specs`, then weight is calculated based on what is packed at `t`.
        - `remain_content` (list[str, ]): Remaining vegetables. If `time_dependent_specs`, then content is calculated based on what is packed at `t`.
        - `count_shifts` (int): How many positions the tray has moved since init.
        - `t_packed` (float): The last time something was packed.

        Args:
            t (float): Current time.
            time_dependent_specs (bool, optional): Allows the content specification to toogle between what is scheduled and what is currently packed. Defaults to False.

        Returns:
            dict: State information.
        """
        remain_weight = self.spec_weight
        remain_content = list(self.spec_content)  # Ensure mutable
        count_shifts = 0
        pos = list(self.pos_init)  # Ensure mutable
        t_packed = None

        for veggie, t_packed in self.packed:
            if time_dependent_specs and t < t_packed:
                continue
            else:
                remain_weight -= veggie.weight
                remain_content.remove(veggie.kind)

        for t_move in self.t_moves:
            if t < t_move:
                break
            pos[0] += self.dist_move
            count_shifts += 1

        return {
            "pos": pos,
            "remain_weight": remain_weight,
            "remain_content": remain_content,
            "count_shifts": count_shifts,
            "t_packed": t_packed,
        }

    def isFull(self, t: float) -> bool:
        """Checks if the tray is full.

        Args:
            t (float): Current time.

        Returns:
            bool
        """
        state = self.getState(t, time_dependent_specs=True)
        return len(state["remain_content"]) == 0

    def hasUnfinishedMoves(self, t: float) -> bool:
        """Checks if the trays is scheduled to move, but has not completed the move yet.

        Args:
            t (float): Current time.

        Returns:
            bool
        """
        if len(self.t_moves) > 0:
            return t < self.t_moves[-1]
        else:
            return False

    def scheduleMove(self, t: float):
        """Schedules a move on the tray.

        Args:
            t (float): Current time.
        """
        # We make sure the list is sorted. This might speed up other calculations.
        for i, time in enumerate(self.t_moves):
            if t < time:
                self.t_moves.insert(i, t)
        else:
            self.t_moves.append(t)

    def canPack(self, veggie: Veggie, t: float = None) -> bool:
        """Check if a given veggie can fit into the tray, given what has been scheduled to be packed.

        Args:
            veggie (Veggie): Vegetable.
            t (float, optional): Time. This is generally not utilized. Defaults to None.

        Returns:
            bool: Result.
        """
        state = self.getState(t=t)
        result = True
        if not veggie.kind in state["remain_content"]:
            result = False

        if len(state["remain_content"]) == 1 and veggie.weight < state["remain_weight"]:
            result = False

        return result

    def packVeggie(self, veggie: Veggie, t: float):
        """Packs a veggie in the tray.

        Args:
            veggie (Veggie): The veggie to pack.
            t (float): Current time.
        """
        if not self.canPack(veggie=veggie, t=t):
            self.canPack(veggie=veggie, t=t)
            raise Exception(f"{veggie} cannot be packed into {self}!")

        # We make sure the list is sorted. This might speed up other calculations.
        i = 0
        for i, (_, time) in enumerate(self.packed):
            if t < time:
                break
        self.packed.insert(i, (veggie, t))
