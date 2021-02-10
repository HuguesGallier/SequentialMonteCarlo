from .FearnheadTaylor import FearnheadTaylor
from .HSMC import HMSC
from .MALASMC import MALASMC
from .Pretuning import Pretuning
from .TunedISMC import TunedISMC

dict_samplers = {"FearnheadTaylor": FearnheadTaylor,
                 "HMSC": HMSC,
                 "MALASMC": MALASMC,
                 "Pretuning": Pretuning,
                 "TunedISMC": TunedISMC}

__all__ = ["FearnheadTaylor", "HMSC", "MALASMC", "Pretuning", "TunedISMC",
           "dict_samplers"]
