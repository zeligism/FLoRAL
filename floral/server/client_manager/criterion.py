from flwr.common.typing import Code, GetPropertiesIns, GetPropertiesRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class MembershipCriterion(Criterion):
    def __init__(self, client_ids) -> None:
        self.client_ids = client_ids

    def select(self, client: ClientProxy) -> bool:
        return client.cid in self.client_ids


class QueryCriterion(Criterion):
    query = "has_something"

    def select(self, client: ClientProxy) -> bool:
        # XXX: this is supposed to be better, but it's incredibly slow in simulation
        ins = GetPropertiesIns({self.query: False})
        res: GetPropertiesRes = client.get_properties(ins, None, None)
        return res.status.code == Code.OK and res.properties[self.query]


class FitCriterion(QueryCriterion):
    query = "has_train"


class EvaluateCriterion(QueryCriterion):
    query = "has_test"
