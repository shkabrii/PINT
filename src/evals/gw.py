from evals.base import Evaluator


class GWEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("GW", eval_cfg, **kwargs)
