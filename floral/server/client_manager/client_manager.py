import random
from flwr.server.client_manager import SimpleClientManager
from .criterion import FitCriterion, EvaluateCriterion, MembershipCriterion


class ClientManagerBase(SimpleClientManager):
    def __init__(self, fit_clients=None, evaluate_clients=None, use_fit_eval_criterion=False) -> None:
        super().__init__()
        self._clients = self.clients
        self.fit_clients = fit_clients
        self.evaluate_clients = evaluate_clients
        self.use_fit_eval_criterion = use_fit_eval_criterion
        self.criterion = None
        self.criterion_per_phase = {}
        if use_fit_eval_criterion:
            self.criterion_per_phase = {
                "fit": FitCriterion() if fit_clients is None else MembershipCriterion(fit_clients),
                "evaluate": EvaluateCriterion() if evaluate_clients is None else MembershipCriterion(evaluate_clients),
            }

    def set_phase(self, phase: str = "fit") -> None:
        if self.use_fit_eval_criterion:
            self.criterion = self.criterion_per_phase.get(phase)
        else:
            if phase == "fit" and self.fit_clients is not None:
                self.clients = {cid: self._clients[cid] for cid in self.fit_clients}
            elif phase == "evaluate" and self.evaluate_clients is not None:
                self.clients = {cid: self._clients[cid] for cid in self.evaluate_clients}
            else:
                self.clients = self._clients

    def sample(self, *args, **kwargs):
        if kwargs.get("criterion") is None:
            kwargs["criterion"] = self.criterion
        return super().sample(*args, **kwargs)
