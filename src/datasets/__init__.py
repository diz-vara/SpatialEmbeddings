from datasets.CityscapesDataset import CityscapesDataset
from datasets.DizInstanceDataset import DizInstanceDataset

def get_dataset(name, dataset_opts):
    if name == "cityscapes": 
        return CityscapesDataset(**dataset_opts)
    elif name == 'diz_instances':
        return DizInstanceDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))