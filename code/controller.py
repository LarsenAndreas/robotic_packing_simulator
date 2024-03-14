import random

from conveyor.arm import Arm
from conveyor.system import System
from conveyor.tray import Tray
from conveyor.veggie import Veggie
from scipy.stats import truncnorm


class Controller:
    def __init__(self, system):
        self.system: System = system

    def __str__(self):
        return f"SimpleController"

    def improbableLastVegetable(self, tray: Tray, t: float, cutoff_prob: float = None):
        data = tray.getState(t)
        remain_veggie = data["remain_content"]
        remain_weight = data["remain_weight"]

        if cutoff_prob and len(remain_veggie) == 1:
            veggie_properties = Veggie.getProperties(remain_veggie[0], rng=None)
            cdf = truncnorm.cdf(
                x=float(remain_weight),
                a=-veggie_properties["clip"],
                b=veggie_properties["clip"],
                loc=float(veggie_properties["mean"]),
                scale=float(veggie_properties["std"]),
            )
            return 1 - cdf < cutoff_prob
        else:
            return False

    def schedulePossibleTrays(self, system_state: dict, cutoff_prob: float = None, force_count: float = 4):
        """Schedule a tray movement, if the last tray in line is filled.
        The movevement is scheduled as to not interferece with the schedules of the arms.

        Args:
            system_state (dict): The state of the system.
        """
        forced = False

        if not cutoff_prob:
            cutoff_prob = self.system.cutoff_probability

        top_trays = sorted([tray for tray in system_state["trays"] if tray.loc == "top"], key=lambda x: x.pos_init[0])
        bot_trays = sorted([tray for tray in system_state["trays"] if tray.loc == "bot"], key=lambda x: x.pos_init[0])

        # Count sequential filled trays
        count_filled_top = 0
        for tray in reversed(top_trays):
            if tray.isFull(t=system_state["t"]):
                count_filled_top += 1
            else:
                break
        count_filled_bot = 0
        for tray in reversed(bot_trays):
            if tray.isFull(t=system_state["t"]):
                count_filled_bot += 1
            else:
                break

        # Force move if a large number is filled
        if count_filled_top >= force_count:
            forced = True
            self.system.clearUnfinishedTasks(t=system_state["t"])
            for _ in range(count_filled_top):
                self.system.moveTrays(loc="top", t=system_state["t"])

        if count_filled_bot >= force_count:
            forced = True
            self.system.clearUnfinishedTasks(t=system_state["t"])
            for _ in range(count_filled_bot):
                self.system.moveTrays(loc="bot", t=system_state["t"])

        t_last_task_top = system_state["t"]
        t_last_task_bot = system_state["t"]
        for arm in system_state["arms"]:
            if len(arm.tasks) == 0:
                continue
            for task in reversed(arm.tasks):
                # Top trays
                if task["target_tray"].loc == "top":
                    t_last_task_top = max(task["time_allocated"][1], t_last_task_top)

                # Bottom trays
                elif task["target_tray"].loc == "bot":
                    t_last_task_bot = max(task["time_allocated"][1], t_last_task_bot)

        for tray in reversed(top_trays):
            is_last_improbable = self.improbableLastVegetable(cutoff_prob=cutoff_prob, tray=tray, t=system_state["t"])
            # is_last_improbable = False
            if (tray.isFull(t=system_state["t"]) or is_last_improbable) and not tray.hasUnfinishedMoves(system_state["t"]):
                if is_last_improbable:
                    tray.discarded = t_last_task_top
                self.system.moveTrays(loc="top", t=t_last_task_top)
            else:
                break

        for tray in reversed(bot_trays):
            is_last_improbable = self.improbableLastVegetable(cutoff_prob=cutoff_prob, tray=tray, t=system_state["t"])
            # is_last_improbable = False
            if (tray.isFull(t=system_state["t"]) or is_last_improbable) and not tray.hasUnfinishedMoves(system_state["t"]):
                if is_last_improbable:
                    tray.discarded = t_last_task_bot
                self.system.moveTrays(loc="bot", t=t_last_task_bot)
            else:
                break

        return forced

    def _validDecision(self, system_state: dict, veggie: Veggie, tray: Tray, arm: Arm) -> dict | None:
        if veggie.t_picked:
            return None

        move_veggie = self.system.getTimeInterceptVeggie(t=system_state["t"], veggie=veggie, arm=arm, system_state=system_state)
        if not move_veggie:
            return None

        t_init_veggie = move_veggie["t_move_start"]
        t_ter_veggie = move_veggie["t_move_end"]

        # Is the arms sphere of influence after the current position of the veggie
        pos_ter_veggie = move_veggie["pos_end"]
        if not arm.canReach(pos_ter_veggie) and not arm.beforeReach(pos_ter_veggie):
            return None

        if not tray.canPack(veggie, t_ter_veggie):
            return None

        move_tray = self.system.getTimeInterceptTray(t=t_ter_veggie, tray=tray, arm=arm, pos_arm=pos_ter_veggie)
        if not move_tray:
            return None
        t_init_tray, dt_tray, t_ter_tray = move_tray

        if t_ter_veggie != t_init_tray:
            return None

        pos_init_arm = arm.getState(t_init_veggie)["pos"]
        pos_ter_tray = tray._getPos(t_ter_tray)
        proposed_movement = [
            {
                "time_start": t_init_veggie,
                "time_stop": t_ter_veggie,
                "pos_start": pos_init_arm,
                "pos_stop": pos_ter_veggie,
            },
            {
                "time_start": t_init_tray,
                "time_stop": t_ter_tray,
                "pos_start": pos_ter_veggie,
                "pos_stop": pos_ter_tray,
            },
        ]

        if len(arm.tasks) == 0:
            # Nothing has been scheduled
            return {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
        else:
            t_prev = t_init_veggie
            for i, task in enumerate(arm.tasks):
                t_init_task, t_ter_task = task["time_allocated"]
                # if (t_init_task <= t_init_veggie <= t_ter_task) or (t_init_task <= t_ter_tray <= t_ter_task):
                if (t_init_veggie <= t_init_task <= t_ter_tray) or (t_init_veggie <= t_ter_task <= t_ter_tray):
                    # Proposed movement is set in existing task
                    return None
                elif (t_prev < t_init_tray) and (t_ter_tray < t_init_task):
                    # Proposed movement fits between tasks. Because of existing tasks, we need to return to starting position
                    dt_return = arm.timeMove(pos_start=tray._getPos(t_ter_tray), pos_stop=arm.getState(t_init_veggie)["pos"])
                    if t_ter_tray + dt_return < t_init_task:
                        # Proposed movement fits between tasks
                        proposed_movement.append(
                            {
                                "time_start": t_ter_tray,
                                "time_stop": t_ter_tray + dt_return,
                                "pos_start": pos_ter_tray,
                                "pos_stop": pos_init_arm,
                            }
                        )
                        return {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
                    break

                elif (i + 1 == len(arm.tasks)) and (t_ter_task < t_init_tray):
                    # The proposed movement is set after the final task
                    return {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
                else:
                    t_prev = t_ter_task

    def _calcFeasibleDecisions(self, system_state: dict):
        """Calculates feasible decisions to a veggie given the system state. It is much faster to restate _validDecision than to call it on every combination of tray, arm, and veggie.

        Args:
            system_state (dict): The state of the system.
        """

        feasible_decisions = []

        for arm in system_state["arms"]:
            for veggie in system_state["veggies"]:
                if veggie.t_picked:
                    continue

                move_veggie = self.system.getTimeInterceptVeggie(t=system_state["t"], veggie=veggie, arm=arm, system_state=system_state)
                if not move_veggie:
                    continue

                t_init_veggie = move_veggie["t_move_start"]
                t_ter_veggie = move_veggie["t_move_end"]

                # Is the arms sphere of influence after the current position of the veggie
                pos_ter_veggie = move_veggie["pos_end"]
                if not arm.canReach(pos_ter_veggie) and not arm.beforeReach(pos_ter_veggie):
                    continue

                for tray in system_state["trays"]:
                    if not tray.canPack(veggie, t_ter_veggie):
                        continue

                    move_tray = self.system.getTimeInterceptTray(t=t_ter_veggie, tray=tray, arm=arm, pos_arm=pos_ter_veggie)
                    if not move_tray:
                        continue
                    t_init_tray, dt_tray, t_ter_tray = move_tray

                    if t_ter_veggie != t_init_tray:
                        continue

                    pos_init_arm = arm.getState(t_init_veggie)["pos"]
                    pos_ter_tray = tray._getPos(t_ter_tray)
                    proposed_movement = [
                        {
                            "time_start": t_init_veggie,
                            "time_stop": t_ter_veggie,
                            "pos_start": pos_init_arm,
                            "pos_stop": pos_ter_veggie,
                        },
                        {
                            "time_start": t_init_tray,
                            "time_stop": t_ter_tray,
                            "pos_start": pos_ter_veggie,
                            "pos_stop": pos_ter_tray,
                        },
                    ]

                    if len(arm.tasks) == 0:
                        # Nothing has been scheduled
                        decision = {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
                        feasible_decisions.append(decision)
                        continue
                    else:
                        t_prev = t_init_veggie
                        for i, task in enumerate(arm.tasks):
                            t_init_task, t_ter_task = task["time_allocated"]
                            if (t_init_veggie <= t_init_task <= t_ter_tray) or (t_init_veggie <= t_ter_task <= t_ter_tray):
                                # Proposed movement is set in existing task
                                break
                            elif (t_prev < t_init_tray) and (t_ter_tray < t_init_task):
                                # Proposed movement fits between tasks. Because of existing tasks, we need to return to starting position
                                dt_return = arm.timeMove(pos_start=tray._getPos(t_ter_tray), pos_stop=arm.getState(t_init_veggie)["pos"])
                                if t_ter_tray + dt_return < t_init_task:
                                    # Proposed movement fits between tasks
                                    proposed_movement.append(
                                        {
                                            "time_start": t_ter_tray,
                                            "time_stop": t_ter_tray + dt_return,
                                            "pos_start": pos_ter_tray,
                                            "pos_stop": pos_init_arm,
                                        }
                                    )
                                    decision = {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
                                    feasible_decisions.append(decision)
                                break
                            elif (i + 1 == len(arm.tasks)) and (t_ter_task < t_init_tray):
                                # The proposed movement is set after the final task
                                decision = {"movement": proposed_movement, "tray": tray, "veggie": veggie, "arm": arm}
                                feasible_decisions.append(decision)
                            else:
                                t_prev = t_ter_task

        return feasible_decisions

    def _randomNaiveController(self) -> float:
        rng = self.system.rng
        random_number = rng.normal(loc=0, scale=1, size=None)
        return random_number

    def _probWeightController(self, tray: Tray, veggie: Veggie, system_state: dict, prio_last: tuple = ("None", 1)) -> float:
        """Given average veggie 'u' and actual veggie 'v'. Let 'w' denote the sum of average weights for the vegetables remaining in a tray.
        The function calculates the distance (u,w) and (v,w), and returns the improvement of packing 'v' compared to 'u'. If negative, then 'u' is considered better than 'v'.

        Args:
            tray (Tray): The tray to pack the veggie in
            veggie (Vegetable): The veggie we are trying to pack
            system_state (dict): The state of the system.

        Returns:
            float: _description_
        """

        t = system_state["t"]
        state_tray = system_state["trays"][tray]
        remain_weight = state_tray["remain_weight"]
        remain_content = state_tray["remain_content"]

        # The expected weight of remaining veggies, given we only see average veggies
        weight_expected = sum([Veggie.getProperties(kind=kind)["mean"] for kind in remain_content])

        overshoot_expected = weight_expected - remain_weight
        overshoot_veggie = weight_expected - Veggie.getProperties(veggie.kind)["mean"] + veggie.weight - remain_weight  # Average weight replaced by actual weight

        unweighted_improvement = abs(overshoot_expected) - abs(overshoot_veggie)

        trays_top = sorted([tray for tray in system_state["trays"] if tray.loc == "top"], key=lambda x: x.pos_init[0])
        trays_bot = sorted([tray for tray in system_state["trays"] if tray.loc == "bot"], key=lambda x: x.pos_init[0])
        if prio_last[0].lower() == "linear":
            if tray.loc == "top":
                a = prio_last[1] / len(trays_top)
                prio_coefficient = a * (trays_top.index(tray) + 1)
            elif tray.loc == "bot":
                a = prio_last[1] / len(trays_bot)
                prio_coefficient = a * (trays_bot.index(tray) + 1)

            if unweighted_improvement > 0:
                improvement = prio_coefficient * (unweighted_improvement)
            else:
                improvement = (unweighted_improvement) / prio_coefficient

        elif prio_last[0].lower() == "exp":
            b = prio_last[1]
            if tray.loc == "top":
                prio_coefficient = b ** (trays_top.index(tray) + 1)
            elif tray.loc == "bot":
                prio_coefficient = b ** (trays_bot.index(tray) + 1)

            if unweighted_improvement > 0:
                improvement = prio_coefficient * (unweighted_improvement)
            else:
                improvement = (unweighted_improvement) / prio_coefficient

        elif prio_last[0].lower() == "none":
            improvement = unweighted_improvement
        return improvement

    def makeDecision(self, system_state: dict, type_controller: str, prio_last: tuple = ("None", 1)):
        if len(system_state["veggies"]) == 0 or len(system_state["trays"]) == 0:
            return None

        match type_controller.lower():
            case "true_random":
                for arm in system_state["arms"]:
                    tray = random.choice(list(system_state["trays"].keys()))
                    veggie = random.choice(list(system_state["veggies"].keys()))
                    decision = self._validDecision(system_state=system_state, veggie=veggie, tray=tray, arm=arm)
                    if decision:
                        decision.pop("arm")
                        arm.scheduleTask(**decision)
            case "random":
                feasible_decisions = self._calcFeasibleDecisions(system_state=system_state)
                for _ in system_state["arms"]:
                    if len(feasible_decisions) == 0:
                        break
                    decision = random.choice(feasible_decisions)
                    arm = decision.pop("arm")
                    arm.scheduleTask(**decision)
                    feasible_decisions = [fd for fd in feasible_decisions if decision["tray"] != fd["tray"] and arm != fd["arm"] and decision["veggie"] != fd["veggie"]]

            case "random_mod":
                feasible_decisions = self._calcFeasibleDecisions(system_state=system_state)
                if len(feasible_decisions) != 0:
                    decision = random.choice(feasible_decisions)
                    arm = decision.pop("arm")
                    arm.scheduleTask(**decision)

            case "prob":
                feasible_decisions = self._calcFeasibleDecisions(system_state=system_state)
                scores = []
                for fd in feasible_decisions:
                    s = self._probWeightController(tray=fd["tray"], veggie=fd["veggie"], system_state=system_state, prio_last=prio_last)
                    scores.append((s, fd))

                # Schedule something on every arm
                for _ in range(len(system_state["arms"])):
                    n_decisions = len(scores)
                    if n_decisions != 0:
                        # Find the best decision
                        score, decision = max(scores, key=lambda x: x[0])

                        # Schedule the decision
                        arm = decision.pop("arm")
                        arm.scheduleTask(**decision)

                    # Remove all decisions which needs to be recalculated
                    scores = [(s, fd) for (s, fd) in scores if decision["tray"] != fd["tray"] and arm != fd["arm"] and decision["veggie"] != fd["veggie"]]

            case "prob_mod":
                feasible_decisions = self._calcFeasibleDecisions(system_state=system_state)
                scores = []
                for fd in feasible_decisions:
                    s = self._probWeightController(tray=fd["tray"], veggie=fd["veggie"], system_state=system_state, prio_last=prio_last)
                    scores.append((s, fd))

                if len(scores) != 0:
                    # Find the best decision
                    _, decision = max(scores, key=lambda x: x[0])

                    # Schedule the decision
                    arm = decision.pop("arm")
                    arm.scheduleTask(**decision)
