# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from sbi.inference.trainers.npe.mnpe import MNPE  # noqa: F401
from sbi.inference.trainers.npe.npe_a import NPE_A  # noqa: F401
from sbi.inference.trainers.npe.npe_b import NPE_B  # noqa: F401
from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer  # noqa: F401
from sbi.inference.trainers.npe.npe_c import NPE_C  # noqa: F401

SNPE_A = NPE_A
SNPE_B = NPE_C
SNPE_C = SNPE = NPE = NPE_C
