from models.BranchedERFNet import BranchedERFNet
from models.BiSeNet import BiSeNet

def get_model(name, model_opts):
    print(name, model_opts)
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    elif name == "BiSeNet":
        model = BiSeNet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))
