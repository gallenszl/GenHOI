import torch

def set_attn_gate(pipe, train=False, model_path=None):
    for block in pipe.dit.blocks:
        block.self_attn.init_lora(train)
        block.self_attn.attn.init_gate(train)
        block.cross_attn.attn.init_gate(train)
        block.cross_attn.attn_img.init_gate(train)
    if model_path is not None:
        print(f"Loading Stand-In weights from: {model_path}")
        load_lora_weights_into_pipe(pipe, model_path)

def load_lora_weights_into_pipe(pipe, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pipe.denoising_model().load_state_dict(ckpt)
    