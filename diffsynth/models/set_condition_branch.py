import torch
from diffsynth import save_video, load_state_dict

def set_stand_in(pipe, train=False, model_path=None, only_gate=False):
    for block in pipe.vace_blocks:
        block.self_attn.attn.init_gate(train)
        block.cross_attn.attn.init_gate(train)
        # block.cross_attn.attn_img.init_gate(train)
    if model_path is not None and only_gate is False:
        print(f"Loading Stand-In weights from: {model_path}")
        load_lora_weights_into_pipe(pipe, model_path)
    if model_path is not None and only_gate:
        print(f"Loading gate weights from: {model_path}")
        state_dict = load_state_dict(model_path)
        state_dict = mapping_gate_state_dict(state_dict)
        pipe.load_state_dict(state_dict, strict=False)
    


def load_lora_weights_into_pipe(pipe, ckpt_path, strict=True):
    ckpt = load_state_dict(ckpt_path)
    pipe.load_state_dict(ckpt)


def set_gate(pipe, train=False, model_path=None):
    for block in pipe.vace_blocks:
        block.self_attn.attn.set_gate_train(train)
        block.cross_attn.attn.set_gate_train(train)
    if model_path is not None:
        print(f"Loading gate weights from: {model_path}")
        state_dict = load_state_dict(model_path)
        state_dict = mapping_gate_state_dict(state_dict)
        pipe.load_state_dict(state_dict, strict=False)

def mapping_gate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "gate" in key:
            new_state_dict[key] = value
    return new_state_dict
