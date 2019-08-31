"""This module provides some configuration for the package."""
import os
import sys

import numpy as np

# We only support modern Python.
np.testing.assert_equal(sys.version_info[0], 3)
np.testing.assert_equal(sys.version_info[1] >= 6, True)


# We rely on relative paths throughout the package.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES_DIR = PACKAGE_DIR + "/test/resources"
