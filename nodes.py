hello world
import os
import gc
import math
import copy
from enum import Enum
from collections import OrderedDict
import folder_paths as comfy_paths
from omegaconf import OmegaConf
import json

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import numpy as np
from safetensors.torch import load_file
from einops import rearrange

from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline
)

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DDIMParallelScheduler,
    LCMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
)

from huggingface_hub import snapshot_download

from plyfile import PlyData
import trimesh
from PIL import Image

from .mesh_processer.mesh import Mesh
from .mesh_processer.mesh_utils import (
    ply_to_points_cloud, 
    get_target_axis_and_scale, 
    switch_ply_axis_and_scale, 
    switch_mesh_axis_and_scale, 
    calculate_max_sh_degree_from_gs_ply,
    marching_cubes_density_to_mesh,
    color_func_to_albedo,
    interpolate_texture_map_attr,
    decimate_mesh,
)

from FlexiCubes.flexicubes_trainer import FlexiCubesTrainer
from DiffRastMesh.diff_mesh import DiffMesh, DiffMeshCameraController
from DiffRastMesh.diff_mesh import DiffRastRenderer
from GaussianSplatting.main_3DGS import GaussianSplatting3D, GaussianSplattingCameraController, GSParams
from GaussianSplatting.main_3DGS_renderer import GaussianSplattingRenderer
from NeRF.Instant_NGP import InstantNGP

from TriplaneGaussian.triplane_gaussian_transformers import TGS
from TriplaneGaussian.utils.config import ExperimentConfig as ExperimentConfigTGS, load_config as load_config_tgs
from TriplaneGaussian.data import CustomImageOrbitDataset
from TriplaneGaussian.utils.misc import todevice, get_device
from LGM.core.options import config_defaults
from LGM.mvdream.pipeline_mvdream import MVDreamPipeline
from LGM.large_multiview_gaussian_model import LargeMultiviewGaussianModel
from LGM.nerf_marching_cubes_converter import GSConverterNeRFMarchingCubes
from TripoSR.system import TSR
from StableFast3D.sf3d import utils as sf3d_utils
from StableFast3D.sf3d.system import SF3D
from InstantMesh.utils.camera_util import oribt_camera_poses_to_input_cameras
from CRM.model.crm.model import ConvolutionalReconstructionModel
from CRM.model.crm.sampler import CRMSampler
from Wonder3D.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from Wonder3D.data.single_image_dataset import SingleImageDataset as MVSingleImageDataset
from Wonder3D.utils.misc import load_config as load_config_wonder3d
from Zero123Plus.pipeline import Zero123PlusPipeline
from Era3D.mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from Era3D.mvdiffusion.data.single_image_dataset import SingleImageDataset as Era3DSingleImageDataset
from Era3D.utils.misc import load_config as load_config_era3d
from Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2mvimg import StableDiffusionImage2MVCustomPipeline
from Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2img import StableDiffusionImageCustomPipeline
from Unique3D.scripts.mesh_init import fast_geo
from Unique3D.scripts.utils import from_py3d_mesh, to_py3d_mesh, to_pyml_mesh, simple_clean_mesh
from Unique3D.scripts.project_mesh import multiview_color_projection, multiview_color_projection_texture, get_cameras_list, get_orbit_cameras_list
from Unique3D.mesh_reconstruction.recon import reconstruct_stage1
from Unique3D.mesh_reconstruction.refine import run_mesh_refine
from CharacterGen.character_inference import Inference2D_API, Inference3D_API
from CharacterGen.Stage_3D.lrm.utils.config import load_config as load_config_cg3d
import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig as ExperimentConfigCraftsman, load_config as load_config_craftsman
from CRM_T2I_V2.model.crm.sampler import CRMSamplerV2
from CRM_T2I_V2.model.t2i_adapter_v2 import T2IAdapterV2
from CRM_T2I_V3.model.crm.sampler import CRMSamplerV3
from Hunyuan3D_V1.mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from Hunyuan3D_V1.mvd.hunyuan3d_mvd_lite_pipeline import Hunyuan3D_MVD_Lite_Pipeline
from Hunyuan3D_V1.infer import Views2Mesh
from Hunyuan3D_V2.hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from Hunyuan3D_V2.hy3dgen.texgen import Hunyuan3DPaintPipeline
from Hunyuan3D_V2.hy3dgen.rembg import BackgroundRemover
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import postprocessing_utils
from TripoSG.pipelines.pipeline_triposg import TripoSGPipeline
from TripoSG.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
from Stable3DGen.pipeline_builders import StableGenPipelineBuilder
from MV_Adapter.mvadapter_node_utils import (
        prepare_pipeline as mvadapter_prepare_pipeline,
        run_pipeline as mvadapter_run_pipeline, 
        prepare_tg2mv_pipeline as mvadapter_prepare_tg2mv_pipeline,
        run_tg2mv_pipeline as mvadapter_run_tg2mv_pipeline,
        prepare_texture_pipeline as mvadapter_prepare_texture_pipeline,
        download_texture_checkpoints,
    )
from mmgp import offload, profile_type
from Gen_3D_Modules.Hunyuan3D_2_1 import (
    FaceReducer_2_1, 
    Hunyuan3DDiTFlowMatchingPipeline_2_1,
    export_to_trimesh_2_1,
    BackgroundRemover_2_1,
    Hunyuan3DPaintPipeline_2_1,
    Hunyuan3DPaintConfig_2_1,
    create_glb_with_pbr_materials_2_1,
)
from Gen_3D_Modules.Hunyuan3D_2_1.hy3dpaint.utils.torchvision_fix import apply_fix
apply_fix()


os.environ['SPCONV_ALGO'] = 'native'

from .shared_utils.image_utils import (
    prepare_torch_img, torch_imgs_to_pils, troch_image_dilate, 
    pils_rgba_to_rgb, pil_make_image_grid, pil_split_image, pils_to_torch_imgs, pils_resize_foreground
)
from .shared_utils.camera_utils import (
    ORBITPOSE_PRESET_DICT, ELEVATION_MIN, ELEVATION_MAX, AZIMUTH_MIN, AZIMUTH_MAX, 
    compose_orbit_camposes
)
from .shared_utils.log_utils import cstr
from .shared_utils.common_utils import parse_save_filename, get_list_filenames, resume_or_download_model_from_hf

DIFFUSERS_PIPE_DICT = OrderedDict([
    ("MVDreamPipeline", MVDreamPipeline),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("Zero123PlusPipeline", Zero123PlusPipeline),
    ("DiffusionPipeline", DiffusionPipeline),
    ("StableDiffusionPipeline", StableDiffusionPipeline),
    ("Era3DPipeline", StableUnCLIPImg2ImgPipeline),
    ("Unique3DImage2MVCustomPipeline", StableDiffusionImage2MVCustomPipeline),
    ("Unique3DImageCustomPipeline", StableDiffusionImageCustomPipeline),
    ("HunYuan3DMVDStdPipeline", HunYuan3D_MVD_Std_Pipeline),
    ("Hunyuan3DMVDLitePipeline", Hunyuan3D_MVD_Lite_Pipeline),
    ("Hunyuan3DDiTFlowMatchingPipeline", Hunyuan3DDiTFlowMatchingPipeline),
    ("Hunyuan3DPaintPipeline", Hunyuan3DPaintPipeline),
    ("TripoSGPipeline", TripoSGPipeline),
    ("TripoSGScribblePipeline", TripoSGScribblePipeline),
])

DIFFUSERS_SCHEDULER_DICT = OrderedDict([
    ("EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("EulerDiscreteScheduler,", EulerDiscreteScheduler),
    ("DDIMScheduler,", DDIMScheduler),
    ("DDIMParallelScheduler,", DDIMParallelScheduler),
    ("LCMScheduler,", LCMScheduler),
    ("KDPM2AncestralDiscreteScheduler,", KDPM2AncestralDiscreteScheduler),
    ("KDPM2DiscreteScheduler,", KDPM2DiscreteScheduler),
])

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CKPT_ROOT_PATH = os.path.join(ROOT_PATH, "Checkpoints")
CKPT_DIFFUSERS_PATH = os.path.join(ROOT_PATH, "Diffusers")
CONFIG_ROOT_PATH = os.path.join(ROOT_PATH, "Configs")
MODULE_ROOT_PATH = os.path.join(ROOT_PATH, "Gen_3D_Modules")

MANIFEST = {
    "name": "ComfyUI-3D-Pack",
    "version": (0,0,2),
    "author": "Mr. For Example",
    "project": "https://github.com/MrForExample/ComfyUI-3D-Pack",
    "description": "An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, etc.)",
}

SUPPORTED_3D_EXTENSIONS = (
    '.obj',
    '.ply',
    '.glb',
)

SUPPORTED_3DGS_EXTENSIONS = (
    '.ply',
)

SUPPORTED_CHECKPOINTS_EXTENSIONS = (
    '.ckpt', 
    '.bin', 
    '.safetensors',
)

WEIGHT_DTYPE = torch.float16

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

HF_DOWNLOAD_IGNORE = ["*.yaml", "*.json", "*.py", ".png", ".jpg", ".gif"]


class Preview_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_gs"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_gs(self, gs_file_path):
        
        gs_folder_path, filename = os.path.split(gs_file_path)
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.output_directory, gs_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3DGS file extensions: {SUPPORTED_3DGS_EXTENSIONS}").error.print()
            gs_file_path = ""
        
        previews = [
            {
                "filepath": gs_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}
    
class Preview_3DMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_mesh"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_mesh(self, mesh_file_path):
        
        mesh_folder_path, filename = os.path.split(mesh_file_path)
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.output_directory, mesh_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
            mesh_file_path = ""
        
        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}

class Load_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, resize, renormal, retex, optimizable, clean, resize_bound):
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    mesh = Mesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )
    
class Load_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "load_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_gs(self, gs_file_path):
        gs_ply = None
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.input_directory, gs_file_path)
        
        if os.path.exists(gs_file_path):
            folder, filename = os.path.split(gs_file_path)
            if filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
                gs_ply = PlyData.read(gs_file_path)
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3DGS file extensions: {SUPPORTED_3DGS_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {gs_file_path} does not exist").error.print()
        return (gs_ply, )
    
class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "save_path": ("STRING", {"default": 'Mesh_%Y-%m-%d-%M-%S-%f.glb', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_mesh(self, mesh, save_path):
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3D_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            mesh.write(save_path)

        return (save_path, )
    
class Save_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "save_path": ("STRING", {"default": '3DGS_%Y-%m-%d-%M-%S-%f.ply', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_gs(self, gs_ply, save_path):
        
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3DGS_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            gs_ply.write(save_path)
        
        return (save_path, )

class Image_Add_Pure_Color_Background:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "R": ("INT", {"default": 255, "min": 0, "max": 255}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "image_add_bg"
    CATEGORY = "Comfy3D/Preprocessor"

    def image_add_bg(self, images, masks, R, G, B):
        """
        bg_mask = bg_mask.unsqueeze(3)
        inv_bg_mask = torch.ones_like(bg_mask) - bg_mask
        color = torch.tensor([R, G, B]).to(image.dtype) / 255
        color_bg = color.repeat(bg_mask.shape)
        image = inv_bg_mask * image + bg_mask * color_bg
        """

        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_rgba_to_rgb(image_pils, (R, G, B))

        images = pils_to_torch_imgs(image_pils, images.dtype, images.device)
        return (images,)
    
class Resize_Image_Foreground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "foreground_ratio": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )
    
    FUNCTION = "resize_img_foreground"
    CATEGORY = "Comfy3D/Preprocessor"

    def resize_img_foreground(self, images, masks, foreground_ratio):
        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_resize_foreground(image_pils, foreground_ratio)
        
        images = pils_to_torch_imgs(image_pils, images.dtype, images.device, force_rgb=False)
        images, masks = images[:, :, :, 0:-1], images[:, :, :, -1]
        return (images, masks,)
    
class Make_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "image_grid",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, images, grid_side_num, use_rows):
        pil_image_list = torch_imgs_to_pils(images)

        if use_rows:
            rows = grid_side_num
            clos = None
        else:
            clos = grid_side_num
            rows = None

        image_grid = pil_make_image_grid(pil_image_list, rows, clos)

        image_grid = TF.to_tensor(image_grid).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]

        return (image_grid,)

class Split_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "split_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def split_image_grid(self, image, grid_side_num, use_rows):
        images = []
        for image_pil in torch_imgs_to_pils(image):

            if use_rows:
                rows = grid_side_num
                clos = None
            else:
                clos = grid_side_num
                rows = None

            image_pils = pil_split_image(image_pil, rows, clos)

            images.append(pils_to_torch_imgs(image_pils, image.dtype, image.device))
            
        images = torch.cat(images, dim=0)
        return (images,)

class Get_Masks_From_Normal_Maps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
            },
        }
        
    RETURN_TYPES = (
        "MASK",
    )
    RETURN_NAMES = (
        "normal_masks",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps):
        from Unique3D.scripts.utils import get_normal_map_masks
        pil_normal_list = torch_imgs_to_pils(normal_maps)
        normal_masks = get_normal_map_masks(pil_normal_list)
        normal_masks = torch.stack(normal_masks, dim=0).to(normal_maps.dtype).to(normal_maps.device)
        return (normal_masks,)

class Rotate_Normal_Maps_Horizontally:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
                "normal_masks": ("MASK",),
                "clockwise": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "normal_maps",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps, normal_masks, clockwise):
        rotate_direction = 1 if clockwise is True else -1
        if normal_maps.shape[0] > 1:
            from Unique3D.scripts.utils import rotate_normals_torch
            pil_image_list = torch_imgs_to_pils(normal_maps, normal_masks)
            pil_image_list = rotate_normals_torch(pil_image_list, return_types='pil', rotate_direction=rotate_direction)
            normal_maps = pils_to_torch_imgs(pil_image_list, normal_maps.dtype, normal_maps.device)
        return (normal_maps,)
    
class Fast_Clean_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "apply_smooth": ("BOOLEAN", {"default": True},),
                "smooth_step": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "apply_sub_divide": ("BOOLEAN", {"default": True},),
                "sub_divide_threshold": ("FLOAT", {"default": 0.25, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "clean_mesh"
    CATEGORY = "Comfy3D/Preprocessor"

    def clean_mesh(self, mesh, apply_smooth, smooth_step, apply_sub_divide, sub_divide_threshold):

        meshes = simple_clean_mesh(to_pyml_mesh(mesh.v, mesh.f), apply_smooth=apply_smooth, stepsmoothnum=smooth_step, apply_sub_divide=apply_sub_divide, sub_divide_threshold=sub_divide_threshold).to(DEVICE)
        vertices, faces, _ = from_py3d_mesh(meshes)

        mesh = Mesh(v=vertices, f=faces, device=DEVICE)

        return (mesh,)
    
class Decimate_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target": ("INT", {"default": 5e4, "min": 0, "max": 0xffffffffffffffff}),
                "remesh": ("BOOLEAN", {"default": True},),
                "optimalplacement": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "process_mesh"
    CATEGORY = "Comfy3D/Preprocessor"

    def process_mesh(self, mesh, target, remesh, optimalplacement):
        vertices, faces = decimate_mesh(mesh.v.detach().cpu().numpy(), mesh.f.detach().cpu().numpy(), target, remesh, optimalplacement)
        mesh.v, mesh.f = torch.from_numpy(vertices).to(DEVICE), torch.from_numpy(faces).to(torch.int64).to(DEVICE)
        mesh.auto_normal()
        return (mesh,)

class Switch_3DGS_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "switched_gs_ply",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, gs_ply, axis_x_to, axis_y_to, axis_z_to):
        switched_gs_ply = None
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to])
            switched_gs_ply = switch_ply_axis_and_scale(gs_ply, target_axis, target_scale, coordinate_invert_count)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis").error.print()
        
        return (switched_gs_ply, )
    
class Switch_Mesh_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
                "flip_normal": ("BOOLEAN", {"default": False},),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "switched_mesh",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, mesh, axis_x_to, axis_y_to, axis_z_to, flip_normal, scale):
        
        switched_mesh = None
        
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to], scale)
            switched_mesh = switch_mesh_axis_and_scale(mesh, target_axis, target_scale, flip_normal)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis
