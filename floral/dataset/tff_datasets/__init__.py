try:
    import tensorflow
    import tensorflow_federated
    from .utils import get_tff_data

except ModuleNotFoundError as e:
    def get_tff_data(**_):
        raise ModuleNotFoundError("Task requires tensorflow-federated but it was not found.")
