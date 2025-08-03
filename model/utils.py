from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Core.RWObj import RWObj_Reader, RWObj_CafReader
from OCC.Extend.DataExchange import read_stl_file
import torch

def get_color(index):
    colors = ['red', 'blue', 'green', 'brown', 'pink', 'yellow', 'purple', 'black']
    return colors[index % len(colors)]

# def read_obj_file(filename):
#     obj_reader = RWObj_Reader()
#     obj_reader.ReadFile(filename)
#     obj_reader.TransferRoots()
#     shape = obj_reader.OneShape()
#     return shape

def read_stl_file_occ(filename):   
    shape = read_stl_file(filename)
    return shape

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(filename)
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape

def convert_hf_to_timm_keys(hf_state_dict):
    """
    Convert HuggingFace ViT-MAE key names to timm ViT key names.
    
    Args:
        hf_state_dict: State dict with HuggingFace naming convention
        
    Returns:
        State dict with timm naming convention
    """
    timm_state_dict = {}
    
    # Store qkv components temporarily to combine them
    qkv_weights = {}
    qkv_biases = {}
    
    for key, value in hf_state_dict.items():
        # Skip head weights - we'll replace with our own
        if key.startswith('head.'):
            continue
            
        # Remove vit. prefix if present
        if key.startswith('vit.'):
            key = key[4:]
        
        # Convert embeddings
        if key == 'embeddings.cls_token':
            timm_state_dict['cls_token'] = value
        elif key == 'embeddings.position_embeddings':
            timm_state_dict['pos_embed'] = value
        elif key.startswith('embeddings.patch_embeddings.projection.'):
            new_key = key.replace('embeddings.patch_embeddings.projection.', 'patch_embed.proj.')
            timm_state_dict[new_key] = value
            
        # Convert transformer blocks
        elif key.startswith('encoder.layer.'):
            # Extract layer number
            parts = key.split('.')
            layer_num = int(parts[2])
            
            if 'attention.attention.query.' in key:
                param_type = key.split('.')[-1]  # weight or bias
                if layer_num not in qkv_weights:
                    qkv_weights[layer_num] = {}
                    qkv_biases[layer_num] = {}
                qkv_weights[layer_num]['query_' + param_type] = value
                
            elif 'attention.attention.key.' in key:
                param_type = key.split('.')[-1]
                if layer_num not in qkv_weights:
                    qkv_weights[layer_num] = {}
                    qkv_biases[layer_num] = {}
                qkv_weights[layer_num]['key_' + param_type] = value
                
            elif 'attention.attention.value.' in key:
                param_type = key.split('.')[-1]
                if layer_num not in qkv_weights:
                    qkv_weights[layer_num] = {}
                    qkv_biases[layer_num] = {}
                qkv_weights[layer_num]['value_' + param_type] = value
                
            elif 'attention.output.dense.' in key:
                param_type = key.split('.')[-1]
                timm_key = f'blocks.{layer_num}.attn.proj.{param_type}'
                timm_state_dict[timm_key] = value
                
            elif 'intermediate.dense.' in key:
                param_type = key.split('.')[-1]
                timm_key = f'blocks.{layer_num}.mlp.fc1.{param_type}'
                timm_state_dict[timm_key] = value
                
            elif 'output.dense.' in key:
                param_type = key.split('.')[-1]
                timm_key = f'blocks.{layer_num}.mlp.fc2.{param_type}'
                timm_state_dict[timm_key] = value
                
            elif 'layernorm_before.' in key:
                param_type = key.split('.')[-1]
                timm_key = f'blocks.{layer_num}.norm1.{param_type}'
                timm_state_dict[timm_key] = value
                
            elif 'layernorm_after.' in key:
                param_type = key.split('.')[-1]
                timm_key = f'blocks.{layer_num}.norm2.{param_type}'
                timm_state_dict[timm_key] = value
                
        # Convert final layer norm
        elif key.startswith('layernorm.'):
            param_type = key.split('.')[-1]
            timm_key = f'norm.{param_type}'
            timm_state_dict[timm_key] = value
    
    # Combine qkv weights and biases
    for layer_num in qkv_weights:
        if ('query_weight' in qkv_weights[layer_num] and 
            'key_weight' in qkv_weights[layer_num] and 
            'value_weight' in qkv_weights[layer_num]):
            
            # Concatenate q, k, v weights
            qkv_weight = torch.cat([
                qkv_weights[layer_num]['query_weight'],
                qkv_weights[layer_num]['key_weight'],
                qkv_weights[layer_num]['value_weight']
            ], dim=0)
            timm_state_dict[f'blocks.{layer_num}.attn.qkv.weight'] = qkv_weight
            
            # Concatenate q, k, v biases if they exist
            if ('query_bias' in qkv_weights[layer_num] and 
                'key_bias' in qkv_weights[layer_num] and 
                'value_bias' in qkv_weights[layer_num]):
                
                qkv_bias = torch.cat([
                    qkv_weights[layer_num]['query_bias'],
                    qkv_weights[layer_num]['key_bias'],
                    qkv_weights[layer_num]['value_bias']
                ], dim=0)
                timm_state_dict[f'blocks.{layer_num}.attn.qkv.bias'] = qkv_bias
    
    return timm_state_dict
