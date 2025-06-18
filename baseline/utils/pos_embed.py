import numpy as np

import torch

def interpolate_pos_embed(model, checkpoint_model):
    for key in ['pos_embed_spatial', 'pos_embed']:
        if key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            print("Embedding size of ckpt: %d" % embedding_size)
            try:
                num_patches = model.patch_embed.num_patches
            except AttributeError as err:
                num_patches = model.patch_embed[0].num_patches
            print("Number of patches of model: %d" % num_patches)
            num_extra_tokens = model.pos_embed_spatial.shape[-2] - num_patches
            print("Number of extra tokens of model: %d" % num_extra_tokens)
            # height (== width) for the checkpoint position embedding
            if num_extra_tokens <= 0:
                orig_size = int((pos_embed_checkpoint.shape[-2]) ** 0.5)
            else:
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            print("Original size of ckpt: %d" % orig_size)
            # height (== width) for the new position embedding
            if key == 'pos_embed_spatial':
                # for the spatial position embedding, the new size is the same as the number of patches
                new_size = int(model.pos_embed_spatial.shape[-2] ** 0.5)
            elif key == 'pos_embed':
                    new_size = int(num_patches ** 0.5)
            else:
                raise ValueError("Unknown key: %s" % key)
            print("New size of model: %d" % new_size)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                if num_extra_tokens > 0:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                else:
                    pos_tokens = pos_embed_checkpoint
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                if num_extra_tokens > 0:
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                else:
                    new_pos_embed = pos_tokens
                checkpoint_model[key] = new_pos_embed

