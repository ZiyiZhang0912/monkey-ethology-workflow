"""
笼内猕猴行为分析 Agent：DLC tracking → 规则行为分类 → 学术可视化。
"""

from .config import AgentConfig, load_config
from .agent import EthologyAgent

__version__ = "0.1.0"
__all__ = ["AgentConfig", "EthologyAgent", "load_config", "__version__"]
