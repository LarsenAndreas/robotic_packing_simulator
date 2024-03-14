import numpy as np
from custom_markers.markers import marker_cucumber, marker_eggplant, marker_tomato
from scipy.stats import truncnorm


class Veggie:
    def __init__(self, t_spawn: float, pos_init: tuple[float, float], kind: str, speed: float, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.t_spawn = t_spawn
        self.pos_init = pos_init
        self.kind = kind
        self.speed = speed
        self.mean, self.weight, *_ = self.getProperties(kind=self.kind, rng=self.rng).values()
        self.t_picked = None

    def __str__(self):
        return self.kind

    def _getPos(self, t) -> tuple[float, float]:
        pos_new = None
        if (self.t_picked is None) or (t < self.t_picked):
            pos_new = (self.pos_init[0] + (t - self.t_spawn) * self.speed, self.pos_init[1])
        return pos_new

    @staticmethod
    def getProperties(kind: str, rng=None) -> dict:
        """Gets properties for a specific vegetable. The properties carries information about the following:
        - "mean" (float): The average weight.
        - "std" (float): The standard deviation of the weight.
        - "prime" (int): The prime associated with the vegetable.
        - "weight" (float): The weight of the vegetable.
        - "clip" (float): How far from the mean that the values are truncated.
        - "marker" (tuple): The marker object and its color (only useful for plotting).

        Args:
            kind (str): The kind of vegetable. Must be in ["cucumber", "tomato", "eggplant"].
            rng (np.random.Generator, optional): If set also samples the weight of the vegetable from a normal distribution. If false the weight is simply the mean.

        Returns:
            (dict): Properties.
        """
        match kind:
            case "cucumber":
                weight_mean, weight_std = 250, 1
                clip = 50
                prime = 2
                marker = (marker_cucumber, "green")
            case "tomato":
                weight_mean, weight_std = 50, 1
                clip = 10
                prime = 3
                marker = (marker_tomato, "red")
            case "eggplant":
                weight_mean, weight_std = 100, 1
                clip = 25
                prime = 5
                marker = (marker_eggplant, "purple")
            case _:
                raise Exception(f'Veggie type "{kind}" is unknown!')

        weight = weight_mean if not rng else truncnorm.rvs(a=-clip, b=clip, loc=weight_mean, scale=weight_std, random_state=rng)
        return {
            "mean": weight_mean,
            "weight": weight,
            "std": weight_std,
            "prime": prime,
            "clip": clip,
            "marker": marker,
        }

    def getState(self, t):
        return {"pos": self._getPos(t), "t_picked": self.t_picked, "kind": self.kind}
