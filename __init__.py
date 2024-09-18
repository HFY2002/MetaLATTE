from transformers import AutoConfig, AutoModel

from .configuration import MetaLATTEConfig
from .model import MultitaskProteinModel

AutoConfig.register("metalatte", MetaLATTEConfig)
AutoModel.register(MetaLATTEConfig, MultitaskProteinModel)