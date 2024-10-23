import os
import json
import numpy as np
from local_dataloader import load_data
from llama_call_hf import process_multilist
import random

EXPLORATION_CONSTANT = np.sqrt(2)
EPSILON_CONSTANT = 1e-5

#define all possible samplers

extra_body = {"use_beam_search": True,
"best_of": 1
}
samplers = [{'temperature':0.0, 'extra_body':extra_body}, {'temperature':0.8, 'top_p':0.7}, {'temperature':1.2, 'top_k':20}]