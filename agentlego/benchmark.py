from typing import List, Optional
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from agentlego.tools import BaseTool
from agentlego.types import ImageIO, Annotated, Info

# from transformers import AutoProcessor, AutoModelForCausalLM
# import torch
# import re

# -----------------------------
# Monkey patch for transformers.AutoModel.from_pretrained
import transformers

old_from_pretrained = transformers.AutoModel.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    return old_from_pretrained(*args, **kwargs)

transformers.AutoModel.from_pretrained = patched_from_pretrained

old_causal_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
def patched_causal_from_pretrained(*args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    return old_causal_from_pretrained(*args, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = patched_causal_from_pretrained

# Patch LlavaLlamaForCausalLM if available
if hasattr(transformers, "LlavaLlamaForCausalLM"):
    old_llava_from_pretrained = transformers.LlavaLlamaForCausalLM.from_pretrained
    def patched_llava_from_pretrained(*args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return old_llava_from_pretrained(*args, **kwargs)
    transformers.LlavaLlamaForCausalLM.from_pretrained = patched_llava_from_pretrained
# -----------------------------

class QwenVLInferencer:

    def __init__(self,
                 model='Qwen/Qwen-VL-Chat',
                 revision='f57cfbd358cb56b710d963669ad1bcfb44cdcdd8',
                 fp16=False,
                 device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, revision=revision)

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device or 'auto',
            trust_remote_code=True,
            revision=revision,
            fp16=fp16).eval()

    def __call__(self, image: ImageIO, text: str):
        query = self.tokenizer.from_list_format(
            [dict(image=image.to_path()), dict(text=text)])
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def fetch_all_box_with_ref(self, text) -> List[dict]:
        items = self.tokenizer._fetch_all_box_with_ref(text)
        return items



class ImageDescription(BaseTool):
    default_desc = ('A useful tool that returns a brief '
                    'description of the input image.')

    @require('mmpretrain')
    def __init__(self,
                 model: str = 'llava-7b-v1.5_vqa',
                 device: str = 'cpu',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import VisualQuestionAnsweringInferencer
        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                VisualQuestionAnsweringInferencer,
                model=self.model,
                device=self.device,
            )

    def apply(self, image: ImageIO) -> str:
        image = image.to_array()[:, :, ::-1]
        return self._inferencer(image, 'Describe the image in detail')[0]['pred_answer']


class CountGivenObject(BaseTool):
    default_desc = 'The tool can count the number of a certain object in the image.'

    @require('mmpretrain')
    def __init__(
        self,
        #model: str = 'llava-7b-v1.5_vqa',
        model: str = 'blip-base_3rdparty_caption',
        # model_id: str = 'llava-hf/llava-1.5-7b-hf',
        device: str = 'cpu',
        toolmeta=None
    ):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        #self.model_id = model_id
        self.device = device

    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import VisualQuestionAnsweringInferencer

        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                VisualQuestionAnsweringInferencer,
                model=self.model,
                device=self.device
            )
        # self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
        #     device_map='auto' if self.device == 'cuda' else None,
        #     trust_remote_code=True, 
        #     attn_implementation='eager',
        # )
        # self.model.to(self.device)
        # self.model.eval()

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')],
        bbox: Annotated[Optional[str],
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
    ) -> int:
        import re
        if bbox is None:
            image = image.to_array()[:, :, ::-1]
        else:
            from agentlego.utils import parse_multi_float
            x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
            image = image.to_array()[y1:y2, x1:x2, ::-1]

        res = self._inferencer(image, f'How many {text} are in the image? Reply a digit')[0]['pred_answer']
        res = re.findall(r'\d+', res)
        if len(res) > 0:
            return int(res[0])
        else:
            return 0

        # #run inference
        # inputs = self.processor(images=image, text=f'How many {text} are in the image? Reply a digit', return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.model.generate(**inputs, max_new_tokens=16)
        # res = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # res = re.findall(r'\d+', res)
        # if len(res) > 0:
        #     return int(res[0])
        # else:
        #     return 0

class RegionAttributeDescription(BaseTool):
    default_desc = 'Describe the attribute of a region of the input image.'

    @require('mmpretrain')
    def __init__(self,
                 model: str = 'llava-7b-v1.5_vqa',
                 device: str = 'cpu',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device


    def setup(self):
        from mmengine.registry import DefaultScope
        from mmpretrain.apis import VisualQuestionAnsweringInferencer
        with DefaultScope.overwrite_default_scope('mmpretrain'):
            self._inferencer = load_or_build_object(
                VisualQuestionAnsweringInferencer,
                model=self.model,
                device=self.device
            )

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> str:
        from agentlego.utils import parse_multi_float
        x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
        cropped_image = image.to_array()[y1:y2, x1:x2, ::-1]
        return self._inferencer(cropped_image, f'Describe {attribute} on the image in detail')[0]['pred_answer']
