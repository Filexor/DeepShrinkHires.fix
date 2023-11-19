import gradio
import torch

import modules.devices as devices
import modules.scripts as scripts
import modules.script_callbacks as script_callbacks
import modules.sd_unet as sd_unet
import modules.shared as shared

from ldm.modules.diffusionmodules.util import timestep_embedding as timestep_embedding

class DeepShrinkHiresFixAction():
    def __init__(self, enable: bool, timestep: float, depth: int, scale: float):
        self.enable = enable
        self.timestep = timestep
        self.depth = depth
        self.scale = scale
        pass
    pass

class DeepShrinkHiresFix(scripts.Script):
    deepShrinkHiresFixActions: list[DeepShrinkHiresFixAction] = []
    enableExperimental: bool = False
    experimentalTimestep: float = 900
    experimentalScales: list[float] = []

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
                Enable_1 = gradio.Checkbox(value=True, label="Enable 1")
                Timestep_1 = gradio.Number(value=900, label="Timestep 1")
                Depth_1 = gradio.Number(value=3, label="Block Depth 1", precision=0)
                Scale_1 = gradio.Number(value=2, label="Scale factor 1")
                pass
            with gradio.Row():
                Enable_2 = gradio.Checkbox(value=True, label="Enable 2")
                Timestep_2 = gradio.Number(value=650, label="Timestep 2")
                Depth_2 = gradio.Number(value=3, label="Block Depth 2", precision=0)
                Scale_2 = gradio.Number(value=2, label="Scale factor 2")
                pass
            with gradio.Accordion(label="Advanced Settings", open=False):
                with gradio.Row():
                    Enable_3 = gradio.Checkbox(value=False, label="Enable 3")
                    Timestep_3 = gradio.Number(value=900, label="Timestep 3")
                    Depth_3 = gradio.Number(value=3, label="Block Depth 3", precision=0)
                    Scale_3 = gradio.Number(value=2, label="Scale factor 3")
                    pass
                with gradio.Row():
                    Enable_4 = gradio.Checkbox(value=False, label="Enable 4")
                    Timestep_4 = gradio.Number(value=650, label="Timestep 4")
                    Depth_4 = gradio.Number(value=3, label="Block Depth 4", precision=0)
                    Scale_4 = gradio.Number(value=2, label="Scale factor 4")
                    pass
                with gradio.Row():
                    Enable_5 = gradio.Checkbox(value=False, label="Enable 5")
                    Timestep_5 = gradio.Number(value=900, label="Timestep 5")
                    Depth_5 = gradio.Number(value=3, label="Block Depth 5", precision=0)
                    Scale_5 = gradio.Number(value=2, label="Scale factor 5")
                    pass
                with gradio.Row():
                    Enable_6 = gradio.Checkbox(value=False, label="Enable 6")
                    Timestep_6 = gradio.Number(value=650, label="Timestep 6")
                    Depth_6 = gradio.Number(value=3, label="Block Depth 6", precision=0)
                    Scale_6 = gradio.Number(value=2, label="Scale factor 6")
                    pass
                with gradio.Row():
                    Enable_7 = gradio.Checkbox(value=False, label="Enable 7")
                    Timestep_7 = gradio.Number(value=900, label="Timestep 7")
                    Depth_7 = gradio.Number(value=3, label="Block Depth 7", precision=0)
                    Scale_7 = gradio.Number(value=2, label="Scale factor 7")
                    pass
                with gradio.Row():
                    Enable_8 = gradio.Checkbox(value=False, label="Enable 8")
                    Timestep_8 = gradio.Number(value=650, label="Timestep 8")
                    Depth_8 = gradio.Number(value=3, label="Block Depth 8", precision=0)
                    Scale_8 = gradio.Number(value=2, label="Scale factor 8")
                    pass
                pass
            with gradio.Accordion(label="Experimental Settings", open=False):
                with gradio.Row():
                    Enable_Experimental = gradio.Checkbox(value=False, label="Enable Experimental Mode")
                    Timestep_Experimental = gradio.Number(value=900, label="Timestep")
                    Scale_Experimental = gradio.Textbox(value="1,1,1, 1,1,1, 1,1,1, 1,1,1, 2, 1,1,1, 1,1,1, 1,1,1, 1,1,1", label="Scale Factor List")
                    pass
                pass
            pass
        return [Enable_1, Timestep_1, Depth_1, Scale_1, Enable_2, Timestep_2, Depth_2, Scale_2, Enable_3, Timestep_3, Depth_3, Scale_3, Enable_4, Timestep_4, Depth_4, Scale_4,
                Enable_5, Timestep_5, Depth_5, Scale_5, Enable_6, Timestep_6, Depth_6, Scale_6, Enable_7, Timestep_7, Depth_7, Scale_7, Enable_8, Timestep_8, Depth_8, Scale_8,
                Enable_Experimental, Timestep_Experimental, Scale_Experimental]
        pass

    def process(self, p, *args):
        del DeepShrinkHiresFix.deepShrinkHiresFixActions[:]
        for i in range(8):
            DeepShrinkHiresFix.deepShrinkHiresFixActions.append(DeepShrinkHiresFixAction(args[i*4], args[i*4+1], args[i*4+2], args[i*4+3]))
            pass
        del DeepShrinkHiresFix.experimentalScales[:]
        DeepShrinkHiresFix.enableExperimental = args[8*4]
        DeepShrinkHiresFix.experimentalTimestep = args[8*4+1]
        scaleFactorsTexts: str = args[8*4+2]
        scaleFactorsTextsList = scaleFactorsTexts.split(",")
        for scaleFactorsText in scaleFactorsTextsList:
            DeepShrinkHiresFix.experimentalScales.append(float(scaleFactorsText))
            pass
        pass

    class DeepShrinkHiresFixUNet(sd_unet.SdUnet):
        def __init__(self, _model):
            super().__init__()
            self.model = _model.to(devices.device)
            pass
        def forward(self, x, timesteps, context, y=None, **kwargs):
            assert (y is not None) == (
                self.model.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            ss = []
            t_emb = timestep_embedding(timesteps, self.model.model_channels, repeat_only=False)
            emb = self.model.time_embed(t_emb)

            if self.model.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.model.label_emb(y)

            h = x.type(self.model.dtype)
            depth = 0
            block = 0
            scale = 1
            for module in self.model.input_blocks:
                for action in DeepShrinkHiresFix.deepShrinkHiresFixActions:
                    if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=1/action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                        break
                        pass
                    pass
                if DeepShrinkHiresFix.enableExperimental and timesteps[0] >= DeepShrinkHiresFix.experimentalTimestep:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=scale/DeepShrinkHiresFix.experimentalScales[block], mode="bicubic", align_corners=False).to(h.dtype)
                    scale = DeepShrinkHiresFix.experimentalScales[block]
                    ss.append(scale)
                    pass
                h = module(h, emb, context)
                hs.append(h)
                depth += 1
                block += 1
                pass

            for action in DeepShrinkHiresFix.deepShrinkHiresFixActions:
                if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=1/action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                    break
                    pass
                pass
            if DeepShrinkHiresFix.enableExperimental and timesteps[0] >= DeepShrinkHiresFix.experimentalTimestep:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=scale/DeepShrinkHiresFix.experimentalScales[block], mode="bicubic", align_corners=False).to(h.dtype)
                scale = DeepShrinkHiresFix.experimentalScales[block]
                pass

            h = self.model.middle_block(h, emb, context)

            for action in DeepShrinkHiresFix.deepShrinkHiresFixActions:
                if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                    break
                    pass
                pass
            block += 1

            for module in self.model.output_blocks:
                depth -= 1
                if DeepShrinkHiresFix.enableExperimental and timesteps[0] >= DeepShrinkHiresFix.experimentalTimestep:
                    h = torch.cat([torch.nn.functional.interpolate(h.float(), scale_factor=scale/DeepShrinkHiresFix.experimentalScales[block], mode="bicubic", align_corners=False).to(h.dtype), 
                                   torch.nn.functional.interpolate(hs.pop().float(), scale_factor=ss.pop()/DeepShrinkHiresFix.experimentalScales[block], mode="bicubic", align_corners=False).to(h.dtype)], dim=1)
                    scale = DeepShrinkHiresFix.experimentalScales[block]
                    pass
                else:
                    h = torch.cat([h, hs.pop()], dim=1)
                    pass
                h = module(h, emb, context)
                for action in DeepShrinkHiresFix.deepShrinkHiresFixActions:
                    if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                        break
                        pass
                    pass
                block += 1
                pass
            if DeepShrinkHiresFix.enableExperimental and timesteps[0] >= DeepShrinkHiresFix.experimentalTimestep:
                h = torch.nn.functional.interpolate(h.float(), scale_factor=scale, mode="bicubic", align_corners=False).to(h.dtype)
                pass
            h = h.type(x.dtype)
            if self.model.predict_codebook_ids:
                return self.model.id_predictor(h)
            else:
                return self.model.out(h)
            pass
        pass

    DeepShrinkHiresFixUNetOption = sd_unet.SdUnetOption()
    DeepShrinkHiresFixUNetOption.label = "Deep Shrink Hires.fix"
    DeepShrinkHiresFixUNetOption.create_unet = lambda: DeepShrinkHiresFix.DeepShrinkHiresFixUNet(shared.sd_model.model.diffusion_model)

    pass

script_callbacks.on_list_unets(lambda unets: unets.append(DeepShrinkHiresFix.DeepShrinkHiresFixUNetOption))