class EfficientResult:
    def __init__(self, result=None):
        if result is not None:
            self.pop = result.pop
            self.initial_state = result.problem.original
            self.n_gen = result.algorithm.n_gen
            self.pop_size = result.algorithm.pop_size
            self.n_offsprings = result.algorithm.n_offsprings
            if hasattr(result.problem, "weight"):
                self.weight = result.problem.weight


class HistoryResult(EfficientResult):
    def __init__(self, result=None):
        super().__init__(result)
        if result is not None:
            self.history = result.problem.history
