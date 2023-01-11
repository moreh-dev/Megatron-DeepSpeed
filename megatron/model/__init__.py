# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from .fused_layer_norm import MixedFusedLayerNormEnc as LayerNormEnc
from .fused_layer_norm import MixedFusedLayerNormDec as LayerNormDec

from .distributed import DistributedDataParallel
from .bert_model import BertModel
from .gpt_model import GPTModel, GPTModelPipe
from .t5_model import T5Model, T5ModelPipe
from .language_model import get_language_model
from .module import Float16Module
