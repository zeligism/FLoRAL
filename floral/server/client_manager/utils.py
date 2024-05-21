from .client_manager import ClientManagerBase

def get_client_manager(cfg, fit_clients=None, evaluate_clients=None):
    client_manager_args = {
        "fit_clients": fit_clients,
        "evaluate_clients": evaluate_clients,
    }
    if "use_fit_eval_criterion" in cfg.strategy:
        client_manager_args["use_fit_eval_criterion"] = cfg.strategy.use_fit_eval_criterion
    return ClientManagerBase(**client_manager_args)
