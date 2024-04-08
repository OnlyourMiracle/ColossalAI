'''
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=39.127.241.68
export MASTER_PORT=5000
export LOCAL_RANK=0
'''

import sys
sys.path.append('/home/aistudio/external-libraries')
from transformers import GPT2Config, GPT2Model
import torch

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from colossalai.legacy.zero.gemini.colo_init_context import ColoInitContext



from colossalai.zero import zero_model_wrapper, zero_optim_wrapper

colossalai.launch_from_torch(config={})
device = get_current_device()

gpt_config = GPT2Config(
    n_layer=24,
    n_embd=2048,
    n_head=24
)

with ColoInitContext(device=device):
    model = GPT2Model(gpt_config)

optimizer = HybridAdam(model.parameters(), lr=1e-3)

gemini_config = dict(
    device=device,
    placement_policy='auto',
    pin_memory=True,
    hidden_dim=model.config.n_embd,
)

model = zero_model_wrapper(
    model,
    zero_stage=3,
    gemini_config=gemini_config,
)

optimizer = zero_optim_wrapper(model, optimizer)

