from collections import OrderedDict

import gradio
import torch

import modules.devices as devices
import modules.scripts as scripts
import modules.script_callbacks as script_callbacks
import modules.sd_unet as sd_unet
import modules.shared as shared

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample, ResBlock
from ldm.modules.diffusionmodules.util import timestep_embedding

class DSHFAction():
    def __init__(self, enable: bool, timestep: float, depth: int, scale: float):
        self.enable = enable
        self.timestep = timestep
        self.depth = depth
        self.scale = scale
        pass
    pass

class DSHFExperimantalAction():
    def __init__(self, enable:bool, timestep: float, scales: list[float], in_multipliers: list[float], out_multipliers: list[float], dilations: list[int]):
        self.enable = enable
        self.timestep = timestep
        self.scales = scales
        self.in_multipliers = in_multipliers
        self.out_multipliers = out_multipliers
        self.dilations = dilations
        pass
    pass

class DSHF(scripts.Script):
    dshf_actions: list[DSHFAction] = []
    enableExperimental: bool = False
    conv2d_only: bool = False
    dshf_experimantal_actions: list[DSHFExperimantalAction] = []
    currentScale: float = 1
    currentBlock: int = 0
    currentConv: int = 0
    currentTimestep: float = 1000

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
                    # presets:
                    # 1; 1;1; 1;1; 1; 1;1; 1;1; 2; 2;1; 1;1; 1; 1;1; 1;1; 1;1;
                    # 1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1
                    pass
                with gradio.Row():
                    Enable_Experimental_1 = gradio.Checkbox(value=True, label="Enable 1")
                    Scale_Experimental_1 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Scale Factors List 1", lines=2)
                    pass
                with gradio.Row():
                    Timestep_Experimental_1 = gradio.Number(value=625, label="Timestep 1")
                    Dilation_Experimental_1 = gradio.Textbox(value="1; 1;1; 1;1; 1; 2;2; 2;2; 2; 2;2; 2;2; 2; 2;2; 2;2; 2;2;\n\
2;2; 2;2; 2;2; 2;2;2; 2;2; 2;2; 2;2;2; 2;2; 2;2; 1;1;1; 1;1; 1;1; 1;1; 1", label="Dilation Factors List 1", lines=2)
                    pass
                with gradio.Row():
                    Premultiplier_Experimental_1 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="In-multipriers List 1")
                    Postmultiplier_Experimental_1 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="Out-multipriers List 1")
                    pass

                with gradio.Row():
                    Enable_Experimental_2 = gradio.Checkbox(value=True, label="Enable 2")
                    Scale_Experimental_2 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 2; 2;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Scale Factors List 2", lines=2)
                    pass
                with gradio.Row():
                    Timestep_Experimental_2 = gradio.Number(value=0, label="Timestep 2")
                    Dilation_Experimental_2 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Dilation Factors List 2", lines=2)
                    pass
                with gradio.Row():
                    Premultiplier_Experimental_2 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="In-multipriers List 2")
                    Postmultiplier_Experimental_2 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="Out-multipriers List 2")
                    pass

                with gradio.Row():
                    Enable_Experimental_3 = gradio.Checkbox(value=False, label="Enable 3")
                    Scale_Experimental_3 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 2; 2;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Scale Factors List 3", lines=2)
                    pass
                with gradio.Row():
                    Timestep_Experimental_3 = gradio.Number(value=750, label="Timestep 3")
                    Dilation_Experimental_3 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Dilation Factors List 3", lines=2)
                    pass
                with gradio.Row():
                    Premultiplier_Experimental_3 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="In-multipriers List 3")
                    Postmultiplier_Experimental_3 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="Out-multipriers List 3")
                    pass
                
                with gradio.Row():
                    Enable_Experimental_4 = gradio.Checkbox(value=False, label="Enable 4")
                    Scale_Experimental_4 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 2; 2;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Scale Factors List 4", lines=2)
                    pass
                with gradio.Row():
                    Timestep_Experimental_4 = gradio.Number(value=750, label="Timestep 4")
                    Dilation_Experimental_4 = gradio.Textbox(value="1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1; 1;1; 1;1; 1;1;\n\
1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1;1; 1;1; 1;1; 1;1; 1", label="Dilation Factors List 4", lines=2)
                    pass
                with gradio.Row():
                    Premultiplier_Experimental_4 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="In-multipriers List 4")
                    Postmultiplier_Experimental_4 = gradio.Textbox(value="1;1;1; 1;1;1; 1;1;1; 1;1;1; 1; 1;1;1; 1;1;1; 1;1;1; 1;1;1; 1", label="Out-multipriers List 4")
                    pass

                pass
            pass
        return [Enable_1, Timestep_1, Depth_1, Scale_1, Enable_2, Timestep_2, Depth_2, Scale_2, Enable_3, Timestep_3, Depth_3, Scale_3, Enable_4, Timestep_4, Depth_4, Scale_4,
                Enable_5, Timestep_5, Depth_5, Scale_5, Enable_6, Timestep_6, Depth_6, Scale_6, Enable_7, Timestep_7, Depth_7, Scale_7, Enable_8, Timestep_8, Depth_8, Scale_8,
                Enable_Experimental, 
                Enable_Experimental_1, Timestep_Experimental_1, Scale_Experimental_1, Premultiplier_Experimental_1, Postmultiplier_Experimental_1, Dilation_Experimental_1,
                Enable_Experimental_2, Timestep_Experimental_2, Scale_Experimental_2, Premultiplier_Experimental_2, Postmultiplier_Experimental_2, Dilation_Experimental_2, 
                Enable_Experimental_3, Timestep_Experimental_3, Scale_Experimental_3, Premultiplier_Experimental_3, Postmultiplier_Experimental_3, Dilation_Experimental_3, 
                Enable_Experimental_4, Timestep_Experimental_4, Scale_Experimental_4, Premultiplier_Experimental_4, Postmultiplier_Experimental_4, Dilation_Experimental_4]
        pass

    def process(self, p, *args):
        del DSHF.dshf_actions[:]
        for i in range(8):
            DSHF.dshf_actions.append(DSHFAction(args[i*4], args[i*4+1], args[i*4+2], args[i*4+3]))
            pass
        del DSHF.dshf_experimantal_actions[:]
        DSHF.enableExperimental = args[8*4]
        for i in range(4):
            scaleslist = args[8*4+1+i*6+2].split(";")
            for j, item in enumerate(scaleslist):
                scaleslist[j] = float(eval(item))
                pass
            premultiplierslist = args[8*4+1+i*6+3].split(";")
            for j, item in enumerate(premultiplierslist):
                premultiplierslist[j] = float(eval(item))
                pass
            postmultiplierslist = args[8*4+1+i*6+4].split(";")
            for j, item in enumerate(postmultiplierslist):
                postmultiplierslist[j] = float(eval(item))
                pass
            dilationlist = args[8*4+1+i*6+5].split(";")
            for j, item in enumerate(dilationlist):
                dilationlist[j] = int(eval(item))
                pass
            DSHF.dshf_experimantal_actions.append(DSHFExperimantalAction(args[8*4+1+i*6], args[8*4+1+i*6+1], scaleslist, premultiplierslist, postmultiplierslist, dilationlist))
            pass
        pass

    class DSHF_Scale(torch.nn.Module):
        def __init__(self, conv2D: list[torch.nn.Conv2d], *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conv2D = conv2D
            pass
        def forward(self, h):
            if DSHF.enableExperimental:
                for action in DSHF.dshf_experimantal_actions:
                    if action.enable == True and action.timestep <= DSHF.currentTimestep:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=1/action.scales[DSHF.currentConv], mode="bicubic", align_corners=False).to(h.dtype)
                        self.conv2D[0].dilation = action.dilations[DSHF.currentConv]
                        self.conv2D[0].padding = action.dilations[DSHF.currentConv]
                        break
                        pass
                    pass
                pass
            return h
            pass
        pass

    class DSHF_Unscale(torch.nn.Module):
        def forward(self, h):
            if DSHF.enableExperimental:
                for action in DSHF.dshf_experimantal_actions:
                    if action.enable == True and action.timestep <= DSHF.currentTimestep:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=action.scales[DSHF.currentConv], mode="bicubic", align_corners=False).to(h.dtype)
                        break
                        pass
                    pass
                pass
            DSHF.currentConv += 1
            return h
            pass
        pass

    class DSHF_InMul(torch.nn.Module):
        def forward(self, h: torch.Tensor):
            if DSHF.enableExperimental:
                for action in DSHF.dshf_experimantal_actions:
                    if action.enable == True and action.timestep <= DSHF.currentTimestep and action.in_multipliers[DSHF.currentBlock] != 1:
                        return h.mul(action.in_multipliers[DSHF.currentBlock])
                        pass
                    pass
                pass
            return h
            pass
        pass

    class DSHF_OutMul(torch.nn.Module):
        def forward(self, h: torch.Tensor):
            if DSHF.enableExperimental:
                for action in DSHF.dshf_experimantal_actions:
                    if action.enable == True and action.timestep <= DSHF.currentTimestep and action.out_multipliers[DSHF.currentBlock] != 1:
                        return h.mul(action.out_multipliers[DSHF.currentBlock])
                        pass
                    pass
                pass
            return h
            pass
        pass


    class DeepShrinkHiresFixUNet(sd_unet.SdUnet):
        def __init__(self, _model):
            super().__init__()
            self.model = _model.to(devices.device)

            for i, input_block in enumerate(self.model.input_blocks):
                for j, layer in enumerate(input_block):
                    if isinstance(layer, ResBlock):
                        for k, in_layer in enumerate(layer.in_layers):
                            if isinstance(in_layer, torch.nn.Conv2d):
                                self.model.input_blocks[i][j].in_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([in_layer]), in_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_InMul())
                                pass
                            pass
                        #self.model.input_blocks[i][j].emb_layers.append(DeepShrinkHiresFix.DSHF_Mul())
                        for k, out_layer in enumerate(layer.out_layers):
                            if isinstance(out_layer, torch.nn.Conv2d):
                                self.model.input_blocks[i][j].out_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([out_layer]), out_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_OutMul())
                                pass
                            pass
                        pass
                    elif isinstance(layer, SpatialTransformer):
                        pass
                    else:
                        if isinstance(layer, torch.nn.Conv2d):
                            self.model.input_blocks[i][j] = torch.nn.Sequential(DSHF.DSHF_Scale([layer]), layer, DSHF.DSHF_Unscale())
                            pass
                        if isinstance(layer, Downsample):
                            self.model.input_blocks[i][j].op = torch.nn.Sequential(DSHF.DSHF_Scale([layer.op]), layer.op, DSHF.DSHF_Unscale())
                            pass
                        if isinstance(layer, Upsample):
                            self.model.input_blocks[i][j].conv = torch.nn.Sequential(DSHF.DSHF_Scale([layer.conv]), layer.conv, DSHF.DSHF_Unscale())
                            pass
                        pass
                    pass
                pass

            for j, layer in enumerate(self.model.middle_block):
                if isinstance(layer, ResBlock):
                    for k, in_layer in enumerate(layer.in_layers):
                        if isinstance(in_layer, torch.nn.Conv2d):
                            self.model.middle_block[j].in_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([in_layer]), in_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_InMul())
                            pass
                        pass
                    #self.model.middle_block[j].emb_layers.append(DeepShrinkHiresFix.DSHF_Mul())
                    for k, out_layer in enumerate(layer.out_layers):
                        if isinstance(out_layer, torch.nn.Conv2d):
                            self.model.middle_block[j].out_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([out_layer]), out_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_OutMul())
                            pass
                        pass
                    pass
                elif isinstance(layer, SpatialTransformer):
                    pass
                else:
                    if isinstance(layer, torch.nn.Conv2d):
                        self.model.middle_block[j] = torch.nn.Sequential(DSHF.DSHF_Scale([layer]), layer, DSHF.DSHF_Unscale())
                        pass
                    if isinstance(layer, Downsample):
                        self.model.middle_block[j].op = torch.nn.Sequential(DSHF.DSHF_Scale([layer.op]), layer.op, DSHF.DSHF_Unscale())
                        pass
                    if isinstance(layer, Upsample):
                        self.model.middle_block[j].conv = torch.nn.Sequential(DSHF.DSHF_Scale([layer.conv]), layer.conv, DSHF.DSHF_Unscale())
                        pass
                    pass
                pass

            for i, output_block in enumerate(self.model.output_blocks):
                for j, layer in enumerate(output_block):
                    if isinstance(layer, ResBlock):
                        for k, in_layer in enumerate(layer.in_layers):
                            if isinstance(in_layer, torch.nn.Conv2d):
                                self.model.output_blocks[i][j].in_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([in_layer]), in_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_InMul())
                                pass
                            pass
                        #self.model.output_blocks[i][j].emb_layers.append(DeepShrinkHiresFix.DSHF_Mul())
                        for k, out_layer in enumerate(layer.out_layers):
                            if isinstance(out_layer, torch.nn.Conv2d):
                                self.model.output_blocks[i][j].out_layers[k] = torch.nn.Sequential(DSHF.DSHF_Scale([out_layer]), out_layer, DSHF.DSHF_Unscale(), DSHF.DSHF_OutMul())
                                pass
                            pass
                        pass
                    elif isinstance(layer, SpatialTransformer):
                        pass
                    else:
                        if isinstance(layer, torch.nn.Conv2d):
                            self.model.output_blocks[i][j] = torch.nn.Sequential(DSHF.DSHF_Scale([layer]), layer, DSHF.DSHF_Unscale())
                            pass
                        if isinstance(layer, Downsample):
                            self.model.output_blocks[i][j].op = torch.nn.Sequential(DSHF.DSHF_Scale([layer.op]), layer.op, DSHF.DSHF_Unscale())
                            pass
                        if isinstance(layer, Upsample):
                            self.model.output_blocks[i][j].conv = torch.nn.Sequential(DSHF.DSHF_Scale([layer.conv]), layer.conv, DSHF.DSHF_Unscale())
                            pass
                        pass
                    pass
                pass

            for i, module in enumerate(self.model.out):
                if isinstance(module, torch.nn.Conv2d):
                    self.model.out[i] = torch.nn.Sequential(DSHF.DSHF_Scale([module]), module, DSHF.DSHF_Unscale())
                    pass
                pass

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
            DSHF.currentBlock = 0
            DSHF.currentConv = 0
            DSHF.currentScale = 1
            DSHF.currentTimestep = timesteps[0]
            for module in self.model.input_blocks:
                for action in DSHF.dshf_actions:
                    if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=1/action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                        break
                        pass
                    pass
                h = module(h, emb, context)
                hs.append(h)
                depth += 1
                DSHF.currentBlock += 1
                pass

            for action in DSHF.dshf_actions:
                if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=1/action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                    break
                    pass
                pass

            h = self.model.middle_block(h, emb, context)

            for action in DSHF.dshf_actions:
                if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                    h = torch.nn.functional.interpolate(h.float(), scale_factor=action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                    break
                    pass
                pass
            DSHF.currentBlock += 1

            for module in self.model.output_blocks:
                depth -= 1
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
                for action in DSHF.dshf_actions:
                    if action.enable == True and action.depth == depth and action.timestep < timesteps[0]:
                        h = torch.nn.functional.interpolate(h.float(), scale_factor=action.scale, mode="bicubic", align_corners=False).to(h.dtype)
                        break
                        pass
                    pass
                DSHF.currentBlock += 1
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
    DeepShrinkHiresFixUNetOption.create_unet = lambda: DSHF.DeepShrinkHiresFixUNet(shared.sd_model.model.diffusion_model)

    pass

script_callbacks.on_list_unets(lambda unets: unets.append(DSHF.DeepShrinkHiresFixUNetOption))