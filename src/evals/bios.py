from evals.base import Evaluator


class BiosEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("BIOS", eval_cfg, **kwargs)
