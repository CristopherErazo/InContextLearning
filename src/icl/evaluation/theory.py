import math
import torch
from torch import Tensor
from typing import Union

# ── type alias ────────────────────────────────────────────────────────────────
FloatLike = Union[float, Tensor]


def effective_loss(ord_params: dict[str, FloatLike], r:int, V:int, K:int, beta:float,return_components:bool=False) -> float:
    m = ord_params['m']
    q = ord_params['Q']
    gamma = ord_params['Gamma']

    rho = K / V
    
    h_star = (beta/r**2) * (1/V) * m * q * gamma

    L_trigg = math.log(1+(V-1)*math.exp(-h_star))
    L_eff = rho*L_trigg + (1-rho)*math.log(V)
    if return_components:
        return {
            "L_eff": L_eff,
            "L_trigg": L_trigg,
            "h_star": h_star
        }
    return L_eff

