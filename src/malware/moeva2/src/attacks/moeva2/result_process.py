
class EfficientResult:
    def __init__(self, result=None):
        if result is not None:
            if hasattr(result.problem, "alg"):
                self.alg = result.problem.alg
            self.pop = result.pop
            self.initial_state = result.problem.get_initial_state()
            self.n_gen = result.algorithm.n_gen
            self.pop_size = result.algorithm.pop_size
            self.n_offsprings = result.algorithm.n_offsprings
            self.X = result.X
            self.F = result.F
            self.pareto = result.problem.last_pareto["X"]
            if "get_weights" in dir(result):
                self.weights = result.problem.get_weights()


class HistoryResult(EfficientResult):
    def __init__(self, result=None):
        super().__init__(result)
        if result is not None:
            self.history = result.problem.get_history()
