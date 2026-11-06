from typing import Any, Dict, Optional

from torch import Tensor


ModelOutput = Dict[str, Any]


def build_model_output(
    node_feat: Optional[Tensor] = None,
    node_sh: Optional[Tensor] = None,
    graph_feat: Optional[Tensor] = None,
    graph_sh: Optional[Tensor] = None,
    **kwargs: Any,
) -> ModelOutput:
    output: ModelOutput = {
        "node_feat": node_feat,
        "node_sh": node_sh,
        "graph_feat": graph_feat,
        "graph_sh": graph_sh,
    }
    output.update(kwargs)
    return output


__all__ = ["ModelOutput", "build_model_output"]
