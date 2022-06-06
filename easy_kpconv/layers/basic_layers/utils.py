from typing import Dict, Optional, Union


def check_bias_from_norm_cfg(norm_cfg: Optional[Union[str, Dict]]) -> bool:
    if norm_cfg is None:
        return True
    if isinstance(norm_cfg, dict):
        norm_cfg = norm_cfg["type"]
    return not norm_cfg.startswith("BatchNorm") and not norm_cfg.startswith("InstanceNorm")
