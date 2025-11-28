from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    # Counts how many episodes ended by termination vs truncation

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.terminated_count = 0
        self.truncated_count = 0

    def _on_step(self) -> bool:
        # self.locals has keys from collect_rollouts
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done:
                    # TimeLimit.truncated in info
                    if info.get("TimeLimit.truncated", False):
                        self.truncated_count += 1
                    else:
                        self.terminated_count += 1

        return True

    def _on_rollout_end(self) -> None:
        # Called once at the end of each collect_rollouts() call
        if self.verbose > 0:
            print(
                f"[Rollout] terminated={self.terminated_count}, "
                f"truncated={self.truncated_count}"
            )
        # Reset counters for next rollout
        self.terminated_count = 0
        self.truncated_count = 0
