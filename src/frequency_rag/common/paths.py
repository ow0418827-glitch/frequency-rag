"""源码布局变化时集中维护的路径。"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
