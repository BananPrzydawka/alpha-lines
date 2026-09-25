"""Model selection for KLENT checkpoints and experiments."""

from config import settings
from models.katago import KataGoNet
from models.katago_tf import KataGoTFNet
from models.resnet import ResNet
from models.maia import MaiaNet


MODELS = {
    'katago': KataGoNet,
    'katago_tf': KataGoTFNet,
    'resnet': ResNet,
    'maia': MaiaNet,
}


def build_model(name, options=None):
    try:
        model_class = MODELS[name]
    except KeyError as error:
        raise ValueError(f'Unknown model: {name}') from error
    return model_class(settings[f'{name}_model'] if options is None else options)
