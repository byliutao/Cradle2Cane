from transformers import AutoProcessor, CLIPModel
import torch
from typing import Optional
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput

class MY_CLIP_MODEL(CLIPModel):
    def get_image_embeds(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs.hidden_states[-2] # follow sdxl text encoder, use the second-last layer 
        image_embeds = self.visual_projection(image_embeds)
                
        pooled_output = vision_outputs.pooler_output  # pooled_output
        pooled_embeds = self.visual_projection(pooled_output)
        
        return image_embeds, pooled_embeds

    