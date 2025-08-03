class Config:
    def __init__(self):
        self.run_inference_reverse = False
        # Dataset related parameters
        self.dataset_file = "/data/dataset/cadparser.json"
        self.data_root = "/data/dataset"
        self.json_data = "/data/dataset/json"
        # self.h5_data = "./data/test"
        self.h5_data = "../dataset/f1/h5"
        self.step_path = "/data/dataset_jun/0627/step/"
        # self.step_path = "./compare_models"
        self.max_total_len = 64
        self.img_height = 384
        self.img_width = 384
        self.data_offload_path = "../dataset/f1/h5"
        self.data_offload_root = "../dataset/f1/"
        # self.data_offload_path = "./data/test"
        # self.data_offload_root = "./data"
        self.offload_sketch_start_point = True
        self.use_new_data_format = False  # Toggle between new (separate H5 files) and old (single H5 file) data loading
        self.n_commands = 12
        self.n_args = 24
        self.args_dim = 256

        # Model related parameters
        self.input_dim = 512
        self.model_dim = 256
        self.n_heads = 8
        self.num_layers = 6
        self.ff_dim = 2048
        self.dropout = 0.1
        self.activation = "gelu"
        self.vocab_size = 10000
        self.max_ep = 64
        # Training related parameters
        self.batch_size = 4
        self.lr = 1e-4
        self.epochs = 20
        self.checkpoint_dir = "./checkpoints/e2"
        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }
        self.log_dir = "./logs"
        self.gpu_idx = [2,3]
        self.use_amp = True  # Enable Automatic Mixed Precision for faster training
        self.norm_img = True
        self.norm_use_mean_std_file = False
        self.use_causal_attention_mask = True
        self.use_post_sum_ln = False
        self.use_desired_reward = True
        # Token order in sequence - options: "RSA" (default), "SAR", "RAS"
        self.token_order = "RSA"
        # Warmup related parameters
        self.use_warmup = True  # Whether to use learning rate warmup
        self.warmup_steps = 1000  # Number of steps for warmup
        self.warmup_method = 'cosine'  # Warmup method: 'linear' or 'cosine'
        self.min_lr_ratio = 0.1  # Minimum lr ratio at the start of warmup
        # only use one global pe
        self.unified_pe = False
        self.use_vit_attention_pooling = False
        self.use_cnn_for_sketch = False  # Whether to use CNN encoder for sketch images instead of ViT
        self.use_cnn_for_all_views = False  # Whether to use CNN encoder for all visual tokens (three views and sketches use different CNN objects)
        
        # Fine-tuned ViT model options
        self.use_finetuned_vit_for_gt = False  # Whether to use ViT model fine-tuned for three views to process GT views
        self.use_finetuned_vit_for_sketch = False  # Whether to use ViT model fine-tuned for sketches to process sketches
        self.finetuned_vit_gt_path = "/data/other_pretrained/vit_mae_gt.pth"  # Path to fine-tuned GT view ViT model
        self.finetuned_vit_sketch_path = "/data/other_pretrained/vit_mae_skt.pth"  # Path to fine-tuned sketch ViT model
        
        # Embedding options
        self.use_enhanced_embedding = False  # Toggle between old and new CADEmbedding implementations
        # for layer norm
        self.use_separate_layernorms = True
        self.use_padding_mask = True
        # GT view augmentation parameters for training robustness
        self.use_gt_view_augmentation = True  # Enable random masking of gt_view images during training
        self.gt_view_mask_prob_one = 0.25  # Probability of masking one image in gt_view (50%)
        self.gt_view_mask_prob_two = 0.25  # Probability of masking two images in gt_view (25%)
        # Comparative experiment options
        self.use_prefix_actions_only = False  # Use only prefix tokens and actions for training (exclude visual state and reward tokens)
        # Inference related parameters
        self.checkpoint_path = "./checkpoints/e2/epoch=13-step=15049.ckpt"
        self.result_path = "./test_results/e2/step"
        self.result_path_seq = "./test_results/e2/seq"
        self.result_root = "./test_results/e2"     
        self.use_single_color_sketch = False  # Option to use only one color for sketches during test
        self.sketch_line_thickness = 2  # Controls the thickness of sketch lines during processing (1-5)   
        # Test related parameters
        self.eval_root = "../../eval/e1"
        self.gt_vec_path = "/data/dataset_jun/0627/vec/"
        # Hash-based image similarity options (test phase only)
        self.use_hash_similarity = True  # Whether to use hash similarity instead of IoU during test phase
        self.hash_method = 'phash'  # Hash method selection: 'phash', 'dhash', 'ahash'
        self.hash_size = 16  # Hash size, affects accuracy

