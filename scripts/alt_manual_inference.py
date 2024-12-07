import os
import torch
import transformers
from transformers import BertPreTrainedModel
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers import StableDiffusionPipeline
from transformers import BertPreTrainedModel,BertModel,BertConfig
import torch.nn as nn
import torch
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import XLMRobertaModel
from transformers.activations import ACT2FN
from typing import Optional

from huggingface_hub import login

login("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")

class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=768,pooler_fn='cls',learn_encoder=False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        # self.learn_encoder = learn_encoder

# class RobertaSeriesModelWithTransformation(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
#     base_model_prefix = 'roberta'
#     config_class= XLMRobertaConfig
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = XLMRobertaModel(config)
#         self.transformation = nn.Linear(config.hidden_size, config.projection_dim)
#         self.post_init()
        
#     def get_text_embeds(self,bert_embeds,clip_embeds):
#         return self.merge_head(torch.cat((bert_embeds,clip_embeds)))

#     def set_tokenizer(self, tokenizer):
#         self.tokenizer = tokenizer

#     def forward(self, input_ids: Optional[torch.Tensor] = None) :
#         attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.int64)
#         outputs = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
        
#         projection_state = self.transformation(outputs.last_hidden_state)
        
#         return (projection_state,)
class RobertaSeriesModelWithTransformation(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    base_model_prefix = 'roberta'
    config_class= XLMRobertaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size, config.projection_dim)
        self.base_model = self.roberta  # Add this line
        self.post_init()
        
    def get_text_embeds(self,bert_embeds,clip_embeds):
        return self.merge_head(torch.cat((bert_embeds,clip_embeds)))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        # If attention_mask is not provided, create it based on input_ids
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.int64)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        projection_state = self.transformation(outputs.last_hidden_state)
        
        return projection_state  # Return the tensor directly instead of a tuple

model_path_encoder = "BAAI/RobertaSeriesModelWithTransformation"
model_path_diffusion = "BAAI/AltDiffusion"
device = "cuda"
cache_dir = '/opt/dlami/nvme'

seed = 12345
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path_encoder, use_auth_token=True, cache_dir=cache_dir)
tokenizer.model_max_length = 77

text_encoder = RobertaSeriesModelWithTransformation.from_pretrained(model_path_encoder, use_auth_token=True, cache_dir=cache_dir)
text_encoder.set_tokenizer(tokenizer)
print("text encode loaded")
pipe = StableDiffusionPipeline.from_pretrained(model_path_diffusion,
                                               tokenizer=tokenizer,
                                               text_encoder=text_encoder,
                                               use_auth_token=True,
                                               cache_dir=cache_dir
                                               )
print("diffusion pipeline loaded")
pipe = pipe.to(device)

prompt = "Thirty years old lee evans as a sad 19th century postman. detailed, soft focus, candle light, interesting lights, realistic, oil canvas, character concept art by munkácsy mihály, csók istván, john everett millais, henry meynell rheam, and da vinci"
with torch.no_grad():
    image = pipe(prompt, guidance_scale=7.5).images[0]  
    
image.save("./alt_attr_outputs/3.png")
