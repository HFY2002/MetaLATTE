from transformers import PretrainedConfig

class MetaLATTEConfig(PretrainedConfig):
    model_type = "metalatte"

    def __init__(
        self,
        num_labels=15,
        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        num_layers_to_finetune=2,
        num_linear_layers=3,
        hidden_dim=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.esm_model_name = esm_model_name
        self.num_layers_to_finetune = num_layers_to_finetune
        self.num_linear_layers = num_linear_layers
        self.hidden_dim = hidden_dim
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)