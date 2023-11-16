import gradio
import torch

import modules.devices as devices
import modules.scripts as scripts
import modules.script_callbacks as script_callbacks
import modules.sd_unet as sd_unet
import modules.shared as shared

from ldm.modules.diffusionmodules.util import timestep_embedding as timestep_embedding

class LatentShrink(scripts.Script):
    timestep_1: float
    depth_1: int
    scale_1: int
    timestep_2: float
    depth_2: int
    scale_2: int

    def __init__(self):
        pass

    def title(self):
        return "Deep Shrink Hires.fix"
        pass

    def show(self, is_img2img):
        return scripts.AlwaysVisible
        pass

    def ui(self, is_img2img):
        with gradio.Accordion(label="Deep Shrink Hires.fix", open=False):
            with gradio.Row():
                Timestep_1 = gradio.Number(value=900, label="Timestep 1")
                Depth_1 = gradio.Number(value=3, label="Block Depth 1", precision=0)
                Scale_1 = gradio.Number(value=2, label="Scale factor 1", precision=0)
                pass
            with gradio.Row():
                Timestep_2 = gradio.Number(value=650, label="Timestep 2")
                Depth_2 = gradio.Number(value=3, label="Block Depth 2", precision=0)
                Scale_2 = gradio.Number(value=2, label="Scale factor 2", precision=0)
                pass
            pass
        return [Timestep_1, Depth_1, Scale_1, Timestep_2, Depth_2, Scale_2]
        pass

    def process(self, p, *args):
        LatentShrink.timestep_1 = args[0]
        LatentShrink.depth_1 = args[1]
        LatentShrink.scale_1 = args[2]
        LatentShrink.timestep_2 = args[3]
        LatentShrink.depth_2 = args[4]
        LatentShrink.scale_2 = args[5]
        pass

    class LatentShrinkUNet(sd_unet.SdUnet):
        def __init__(self, _model):
            super().__init__()
            self.model = _model.to(devices.device)
            pass
        def forward(self, x, timesteps, context, y=None, **kwargs):
            assert (y is not None) == (
                self.model.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, self.model.model_channels, repeat_only=False)
            emb = self.model.time_embed(t_emb)

            if self.model.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.model.label_emb(y)

            h = x.type(self.model.dtype)
            depth = 0
            for module in self.model.input_blocks:
                if depth == LatentShrink.depth_1 and timesteps[0] > LatentShrink.timestep_1:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=1/LatentShrink.scale_1, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                    pass
                elif depth == LatentShrink.depth_2 and timesteps[0] > LatentShrink.timestep_2:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=1/LatentShrink.scale_2, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                    pass
                h = module(h, emb, context)
                hs.append(h)
                depth += 1
                pass

            if depth == LatentShrink.depth_1 and timesteps[0] > LatentShrink.timestep_1:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=1/LatentShrink.scale_1, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                pass
            elif depth == LatentShrink.depth_2 and timesteps[0] > LatentShrink.timestep_2:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=1/LatentShrink.scale_2, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                pass
            h = self.model.middle_block(h, emb, context)
            if depth == LatentShrink.depth_1 and timesteps[0] > LatentShrink.timestep_1:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=LatentShrink.scale_1, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                pass
            elif depth == LatentShrink.depth_2 and timesteps[0] > LatentShrink.timestep_2:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=LatentShrink.scale_2, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                pass

            for module in self.model.output_blocks:
                depth -= 1
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
                if depth == LatentShrink.depth_1 and timesteps[0] > LatentShrink.timestep_1:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=LatentShrink.scale_1, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                    pass
                elif depth == LatentShrink.depth_2 and timesteps[0] > LatentShrink.timestep_2:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=LatentShrink.scale_2, mode="bicubic", align_corners=False).to(h.dtype)  # bfloat16対応
                    pass
                pass
            h = h.type(x.dtype)
            if self.model.predict_codebook_ids:
                return self.model.id_predictor(h)
            else:
                return self.model.out(h)
            pass
        pass

    LatentShrinkUNetOption = sd_unet.SdUnetOption()
    LatentShrinkUNetOption.label = "Latent Shrink"
    LatentShrinkUNetOption.create_unet = lambda: LatentShrink.LatentShrinkUNet(shared.sd_model.model.diffusion_model)

    pass

script_callbacks.on_list_unets(lambda unets: unets.append(LatentShrink.LatentShrinkUNetOption))