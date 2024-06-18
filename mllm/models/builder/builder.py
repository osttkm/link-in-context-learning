from torch import nn
from typing import Dict, Any, Tuple
from .build_llava import load_pretrained_llava

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if type_ == 'oshita_llava':
        return load_pretrained_llava(model_args, training_args)
    else:
        assert False
