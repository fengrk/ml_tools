# -*- coding:utf-8 -*-
from __future__ import absolute_import

try:
    from pyxtools.faiss_tools import faiss, IndexType, FaissManager, ImageIndexUtils
except ImportError:
    from pyxtools.pyxtools.faiss_tools import faiss, IndexType, FaissManager, ImageIndexUtils
