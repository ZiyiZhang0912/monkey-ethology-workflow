"""
笼内猕猴行为分析 Workflow：DLC tracking → 规则行为分类 → 学术可视化。
"""

from .config import WorkflowConfig, load_config
from .workflow import EthologyWorkflow

__version__ = "0.1.0"
__all__ = ["WorkflowConfig", "EthologyWorkflow", "load_config", "__version__"]
