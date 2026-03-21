"""IDLoraModelLoader nodes — loads checkpoint + text encoder + LoRA into pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from comfy_api.latest import io
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP

from .pipeline_wrapper import IDLoraOneStagePipeline, IDLoraTwoStagePipeline, IDLoraPipelineType


# Base directory for resolving relative model paths.
# ComfyUI may change CWD, so we anchor to the repository root (two levels
# above this file: custom_nodes/comfyui-id-lora-ltx/ -> ComfyUI/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(p: str) -> str:
    """Return *p* unchanged if absolute, otherwise resolve relative to the repo root."""
    if not p or os.path.isabs(p):
        return p
    resolved = _REPO_ROOT / p
    return str(resolved) if resolved.exists() else p


class IDLoraModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraModelLoader",
            display_name="ID-LoRA Model Loader",
            category="ID-LoRA",
            description="Load LTX-2.3 checkpoint, text encoder, and ID-LoRA weights into a reusable pipeline.",
            inputs=[
                io.String.Input("checkpoint_path", default="models/ltx-2.3-22b-dev.safetensors",
                                tooltip="Path to the LTX-2.3 base checkpoint (.safetensors)."),
                io.String.Input("text_encoder_path", default="models/gemma-3-12b-it-qat-q4_0-unquantized",
                                tooltip="Path to the Gemma text encoder directory."),
                io.String.Input("lora_path", default="",
                                tooltip="Path to the ID-LoRA checkpoint (.safetensors)."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="LoRA application strength."),
                io.Combo.Input("quantize", options=["none", "int8", "fp8"],
                               tooltip="Quantization mode for the transformer."),
                io.Float.Input("stg_scale", default=1.0, min=0.0, max=10.0, step=0.1,
                               tooltip="STG (Spatio-Temporal Guidance) scale. 0 disables."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Identity guidance scale for speaker transfer."),
                io.Float.Input("av_bimodal_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Audio-video bimodal CFG scale."),
            ],
            outputs=[
                io.Custom("ID_LORA_PIPELINE").Output(display_name="Pipeline", tooltip="Loaded ID-LoRA pipeline."),
            ],
        )

    @classmethod
    def execute(
        cls,
        checkpoint_path: str,
        text_encoder_path: str,
        lora_path: str,
        lora_strength: float,
        quantize: str,
        stg_scale: float,
        identity_guidance_scale: float,
        av_bimodal_scale: float,
    ) -> io.NodeOutput:
        device = torch.device("cuda")
        checkpoint_path = _resolve_path(checkpoint_path)
        text_encoder_path = _resolve_path(text_encoder_path)

        loras = []
        if lora_path.strip():
            loras.append(LoraPathStrengthAndSDOps(
                path=_resolve_path(lora_path.strip()),
                strength=lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))

        pipeline = IDLoraOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=text_encoder_path,
            loras=loras,
            device=device,
            quantize=(quantize == "int8"),
            fp8=(quantize == "fp8"),
            stg_scale=stg_scale,
            identity_guidance=True,
            identity_guidance_scale=identity_guidance_scale,
            av_bimodal_cfg=True,
            av_bimodal_scale=av_bimodal_scale,
        )

        # NOTE: load_models() is NOT called here. The original ID-LoRA script
        # encodes prompts BEFORE loading the heavy models (transformer, VAEs)
        # to avoid GPU memory pressure. The sampler node calls load_models()
        # after prompt encoding is complete.
        return io.NodeOutput(pipeline)


class IDLoraTwoStageModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraTwoStageModelLoader",
            display_name="ID-LoRA Two-Stage Model Loader",
            category="ID-LoRA",
            description="Load LTX-2.3 checkpoint, text encoder, ID-LoRA, upsampler, and distilled LoRA for two-stage generation.",
            inputs=[
                io.String.Input("checkpoint_path", default="models/ltx-2.3-22b-dev.safetensors",
                                tooltip="Path to the LTX-2.3 base checkpoint (.safetensors)."),
                io.String.Input("text_encoder_path", default="models/gemma-3-12b-it-qat-q4_0-unquantized",
                                tooltip="Path to the Gemma text encoder directory."),
                io.String.Input("lora_path", default="",
                                tooltip="Path to the ID-LoRA checkpoint (.safetensors)."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="LoRA application strength."),
                io.String.Input("upsampler_path", default="models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                                tooltip="Path to the spatial upsampler checkpoint (.safetensors)."),
                io.String.Input("distilled_lora_path", default="models/ltx-2.3-22b-distilled-lora-384.safetensors",
                                tooltip="Path to the distilled LoRA for stage 2 (.safetensors)."),
                io.Combo.Input("quantize", options=["none", "int8", "fp8"],
                               tooltip="Quantization mode for the transformer."),
                io.Float.Input("stg_scale", default=1.0, min=0.0, max=10.0, step=0.1,
                               tooltip="STG (Spatio-Temporal Guidance) scale. 0 disables."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Identity guidance scale for speaker transfer."),
                io.Float.Input("av_bimodal_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Audio-video bimodal CFG scale."),
            ],
            outputs=[
                io.Custom("ID_LORA_PIPELINE").Output(display_name="Pipeline", tooltip="Loaded ID-LoRA two-stage pipeline."),
            ],
        )

    @classmethod
    def execute(
        cls,
        checkpoint_path: str,
        text_encoder_path: str,
        lora_path: str,
        lora_strength: float,
        upsampler_path: str,
        distilled_lora_path: str,
        quantize: str,
        stg_scale: float,
        identity_guidance_scale: float,
        av_bimodal_scale: float,
    ) -> io.NodeOutput:
        device = torch.device("cuda")
        checkpoint_path = _resolve_path(checkpoint_path)
        text_encoder_path = _resolve_path(text_encoder_path)
        upsampler_path = _resolve_path(upsampler_path)
        distilled_lora_path = _resolve_path(distilled_lora_path)

        ic_loras = []
        if lora_path.strip():
            ic_loras.append(LoraPathStrengthAndSDOps(
                path=_resolve_path(lora_path.strip()),
                strength=lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))

        pipeline = IDLoraTwoStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=text_encoder_path,
            upsampler_path=upsampler_path,
            distilled_lora_path=distilled_lora_path,
            ic_loras=ic_loras,
            device=device,
            quantize=(quantize == "int8"),
            fp8=(quantize == "fp8"),
            stg_scale=stg_scale,
            identity_guidance=True,
            identity_guidance_scale=identity_guidance_scale,
            av_bimodal_cfg=True,
            av_bimodal_scale=av_bimodal_scale,
        )

        # NOTE: load_stage_1_models() is NOT called here — deferred to sampler
        # node after prompt encoding is complete, to minimize peak VRAM.
        return io.NodeOutput(pipeline)
