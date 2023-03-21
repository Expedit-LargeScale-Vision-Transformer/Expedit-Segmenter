import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='prepocess checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()
    return args

def prepocess(ckpt):
    def _change_key(old_key, new_key):
        if old_key in ckpt['state_dict'].keys():
            ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(old_key)

    if 'meta' not in ckpt.keys():
        ckpt['meta'] = {}
    if 'n_cls' in ckpt.keys():
        ckpt['meta']['n_cls'] = ckpt.pop('n_cls')
    if 'epoch' in ckpt.keys():
        ckpt['meta']['epoch'] = ckpt.pop('epoch')

    if 'model' in ckpt.keys():
        ckpt['state_dict'] = ckpt.pop('model')

    _change_key('encoder.cls_token', 'backbone.cls_token')
    _change_key('encoder.pos_embed', 'backbone.pos_embed')
    _change_key('encoder.patch_embed.proj.weight', 'backbone.patch_embed.projection.weight')
    _change_key('encoder.patch_embed.proj.bias', 'backbone.patch_embed.projection.bias')

    for i in range(24):
        _change_key(f'encoder.blocks.{i}.norm1.weight', f'backbone.layers.{i}.ln1.weight')
        _change_key(f'encoder.blocks.{i}.norm1.bias', f'backbone.layers.{i}.ln1.bias')
        _change_key(f'encoder.blocks.{i}.norm2.weight', f'backbone.layers.{i}.ln2.weight')
        _change_key(f'encoder.blocks.{i}.norm2.bias', f'backbone.layers.{i}.ln2.bias')
        _change_key(f'encoder.blocks.{i}.attn.qkv.weight', f'backbone.layers.{i}.attn.attn.in_proj_weight')
        _change_key(f'encoder.blocks.{i}.attn.qkv.bias', f'backbone.layers.{i}.attn.attn.in_proj_bias')
        _change_key(f'encoder.blocks.{i}.attn.proj.weight', f'backbone.layers.{i}.attn.attn.out_proj.weight')
        _change_key(f'encoder.blocks.{i}.attn.proj.bias', f'backbone.layers.{i}.attn.attn.out_proj.bias')
        _change_key(f'encoder.blocks.{i}.mlp.fc1.weight', f'backbone.layers.{i}.ffn.layers.0.0.weight')
        _change_key(f'encoder.blocks.{i}.mlp.fc1.bias', f'backbone.layers.{i}.ffn.layers.0.0.bias')
        _change_key(f'encoder.blocks.{i}.mlp.fc2.weight', f'backbone.layers.{i}.ffn.layers.1.weight')
        _change_key(f'encoder.blocks.{i}.mlp.fc2.bias', f'backbone.layers.{i}.ffn.layers.1.bias')

        
    _change_key('encoder.norm.weight', 'backbone.ln1.weight')
    _change_key('encoder.norm.bias', 'backbone.ln1.bias')
    # _change_key('encoder.head.weight', 'backbone.ln1.weight')
    # _change_key('encoder.head.bias', 'backbone.ln1.bias')

    for k in list(ckpt['state_dict'].keys()):
        if 'decoder.' in k:
            _change_key(k, k.replace('decoder.', 'decode_head.'))

    return ckpt

def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    print("load " + args.checkpoint)
    ckpt = prepocess(ckpt)
    print("save to " + args.checkpoint)
    torch.save(ckpt, args.checkpoint)
    
if __name__ == '__main__':
    main()