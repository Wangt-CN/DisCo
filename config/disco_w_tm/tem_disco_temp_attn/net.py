from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.utils import PIL_INTERPOLATION, deprecate
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from packaging import version
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from config import *
from config.ref_attn_clip_combine_controlnet.net import Net
from utils.dist import get_rank, synchronize

from .controlnet_3d import ControlNetModel3D, MultiControlNetModel_MultiHiddenStates
from .unet_3d_condition import UNet3DConditionModel


class TemConvNet(Net):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        tr_noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path, subfolder="scheduler"
        )
        if self.args.eval_scheduler == "ddpm":
            noise_scheduler = DDPMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler"
            )
        elif self.args.eval_scheduler == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler"
            )
        else:
            noise_scheduler = PNDMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler"
            )
        # tokenizer = CLIPTokenizer.from_pretrained(
        #     args.pretrained_model_path, subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_path, subfolder="feature_extractor"
        )
        print(
            f"Loading pre-trained image_encoder from {args.pretrained_model_path}/image_encoder"
        )
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_path, subfolder="image_encoder"
        )
        print(f"Loading pre-trained vae from {args.pretrained_model_path}/vae")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        print(f"Loading pre-trained unet from {self.args.pretrained_model_path}/unet")
        unet = UNet3DConditionModel.from_pretrained_2d(
            self.args.pretrained_model_path, subfolder="unet"
        )

        if (
            hasattr(noise_scheduler.config, "steps_offset")
            and noise_scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {noise_scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(self.noise_scheduler.config)
            new_config["steps_offset"] = 1
            noise_scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(noise_scheduler.config, "clip_sample")
            and noise_scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(noise_scheduler.config)
            new_config["clip_sample"] = False
            noise_scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(self.unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")

        # WT: initialize controlnet from the pretrained image variation SD model
        # pose controlnet
        controlnet_list = []
        controlnet_pose = ControlNetModel3D.from_unet_2d(unet=unet, args=self.args)
        controlnet_list.append(controlnet_pose)

        # background controlnet
        if args.ref_null_caption:  # default to False
            raise NotImplementedError
            tokenizer = CLIPTokenizer.from_pretrained(
                self.args.sd15_path, subfolder="tokenizer"
            )
            self.tokenizer = tokenizer
            print(
                f"Loading pre-trained text_encoder from {self.args.sd15_path}/text_encoder"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.args.sd15_path + "/text_encoder"
            )
            self.text_encoder = text_encoder
            unet_sd15 = UNet2DConditionModel.from_pretrained(
                self.args.sd15_path, subfolder="unet"
            )
            controlnet_background = ControlNetModel2D.from_unet(
                unet=unet_sd15, args=self.args, use_sd_vae=True
            )  # initialize ref controlnet path from the SD pretrained model
            del unet_sd15
        else:  # initialize controlnet from the variation SD
            controlnet_background = ControlNetModel3D.from_unet_2d(
                unet=unet, args=self.args, use_sd_vae=True
            )
        controlnet_list.append(controlnet_background)

        if self.args.gradient_checkpointing:
            for i in range(len(controlnet_list)):
                controlnet_list[i].enable_gradient_checkpointing()
        controlnet_unit = MultiControlNetModel_MultiHiddenStates(controlnet_list)

        self.tr_noise_scheduler = tr_noise_scheduler
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.controlnet = controlnet_unit
        self.unet = unet
        self.feature_extractor = feature_extractor
        self.clip_image_encoder = clip_image_encoder
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.drop_text_prob = args.drop_text
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        self.scale_factor = self.args.scale_factor
        self.guidance_scale = self.args.guidance_scale
        self.controlnet_conditioning_scale = getattr(
            self.args, "controlnet_conditioning_scale", 1.0
        )
        self.controlnet_conditioning_scale_cond = getattr(
            self.args, "controlnet_conditioning_scale_cond", 1.0
        )
        self.controlnet_conditioning_scale_ref = getattr(
            self.args, "controlnet_conditioning_scale_ref", 1.0
        )

        if getattr(self.args, "combine_clip_local", None) and not getattr(
            self.args, "refer_clip_proj", None
        ):  # not use clip pretrained visual projection (but initialize from it)
            self.refer_clip_proj = torch.nn.Linear(
                clip_image_encoder.visual_projection.in_features,
                clip_image_encoder.visual_projection.out_features,
                bias=False,
            )
            self.refer_clip_proj.load_state_dict(
                clip_image_encoder.visual_projection.state_dict()
            )
            self.refer_clip_proj.requires_grad_(True)

    def freeze_pretrained_part_in_ddpm(self):
        if getattr(self.args, "stage2_only_pose", False):
            print("Only train the pose path, and freeze all the other params")
            # freeze all the param, only without the pose controlnet
            self.args.freeze_unet = True
            self.args.unet_unfreeze_type = None  # freeze all the

            # for background controlnet
            for param_name, param in self.controlnet.named_parameters():
                if not "nets.0" in param_name:
                    param.requires_grad_(False)

            # for clip projector
            if hasattr(self, "refer_clip_proj"):
                for param_name, param in self.refer_clip_proj.named_parameters():
                    param.requires_grad_(False)

        if self.args.freeze_unet:
            # b self.unet.eval()
            param_unfreeze_num = 0
            if self.args.unet_unfreeze_type == "crossattn-kv":
                for (
                    param_name,
                    param,
                ) in (
                    self.unet.named_parameters()
                ):  # only to set attn2 k, v to be requires_grad
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    elif not ("attn2.to_k" in param_name or "attn2.to_v" in param_name):
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "crossattn":
                for (
                    param_name,
                    param,
                ) in (
                    self.unet.named_parameters()
                ):  # only to set attn2 k, v to be requires_grad
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    elif not "attn2" in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "tem":
                for param_name, param in self.unet.named_parameters():
                    if "tem_" not in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "transblocks":
                for param_name, param in self.unet.named_parameters():
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "all":
                for param_name, param in self.unet.named_parameters():
                    param.requires_grad_(True)
                    param_unfreeze_num += 1

            else:  # freeze all the unet
                print("Unmatch to any option, freeze all the unet")
                self.unet.eval()
                self.unet.requires_grad_(False)

            print(
                f"Mode [{self.args.unet_unfreeze_type}]: There are {param_unfreeze_num} modules in unet to be set as requires_grad=True."
            )

        self.vae.eval()
        self.clip_image_encoder.eval()
        self.vae.requires_grad_(False)
        self.clip_image_encoder.requires_grad_(False)
        if hasattr(self, "text_encoder"):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

    def forward_train_multicontrol(self, inputs, outputs):
        # use CFG
        if self.args.drop_ref > 0:
            p = random.random()
            if p <= self.args.drop_ref:  # dropout ref image
                inputs["reference_img"] = torch.zeros_like(inputs["reference_img"])

        loss_target = self.args.loss_target
        image_seq = inputs["label_img_seq"]  # (B, C, T, H, W)
        ref_image = inputs["reference_img"]
        bsz = image_seq.shape[0]

        # drop pose
        if self.args.drop_pose_ratio > 0:
            for t in range(self.args.nframes):
                p = random.random()
                if p <= self.args.drop_pose_ratio:
                    inputs["cond_img_seq"][:, :, t] = torch.ones_like(
                        inputs["cond_img_seq"][:, :, t]
                    ) * inputs["cond_img_seq"][:, :, t, 0, 0].unsqueeze(2).unsqueeze(2)

        if self.args.ref_null_caption:  # default to False
            text = inputs["input_text"]
            if random.random() < self.args.drop_text:  # drop text w.r.t the prob
                text = ["" for i in text]
            z_text = self.text_encode(text)

        # text SD input --> reference image input (clip global embedding)
        if self.args.combine_clip_local:  # default to True
            refer_latents = self.clip_encode_image_local(ref_image).to(dtype=self.dtype)
        else:
            refer_latents = self.clip_encode_image_global(ref_image).to(
                dtype=self.dtype
            )

        image_seq = rearrange(image_seq, "b c t h w -> (b t) c h w")
        latents = self.image_encoder(image_seq)
        latents = rearrange(latents, "(b t) c h w -> b c t h w", t=self.args.nframes)

        latents = latents.to(dtype=self.dtype)
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.tr_noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        if self.args.debug_seed:
            print(
                f"rank {get_rank()}: noise 0 mean {torch.sum(noise[0])}, noise 1 mean {torch.sum(noise[1])}"
            )
            print(f"timestep 0 {timesteps[0]}, timestep 1 {timesteps[1]}")
        noisy_latents = self.tr_noise_scheduler.add_noise(latents, noise, timesteps)

        # TODO: @tan, change cond_imgs in dataloadser to pose or other conditions.
        # Prepare reference image
        if self.args.refer_sdvae:  # default to True
            reference_latents_controlnet = self.image_encoder(
                inputs["reference_img_controlnet"]
            )  # controlnet path input
            reference_latents_controlnet = reference_latents_controlnet.to(
                dtype=self.dtype
            )
        else:
            reference_latents_controlnet = inputs["reference_img_controlnet"]
        controlnet_image = [
            inputs["cond_img_seq"],
            reference_latents_controlnet,
        ]  # [pose image, ref image]

        # controlnet get the input of (a. ref image clip embedding; b. pose cond image; c. [optional] depth; d. [optional] previous frame)
        if self.args.ref_null_caption:  # default to False
            raise NotImplementedError
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                [refer_latents, z_text],  # reference controlnet path use null caption
                controlnet_cond=controlnet_image,
                conditioning_scale=[1.0, 1.0],
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                [refer_latents]
                * len(self.args.conds),  # both controlnet path use the refer latents
                controlnet_cond=controlnet_image,
                conditioning_scale=[1.0, 1.0],
                nframes=self.args.nframes,
                return_dict=False,
            )

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=refer_latents,  # refer latents
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        if loss_target == "x0":
            target = latents
            x0_pred = self.tr_noise_scheduler.remove_noise(
                noisy_latents, model_pred, timesteps
            )
            loss = F.mse_loss(x0_pred.float(), target.float(), reduction="mean")
        else:
            if self.tr_noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif self.tr_noise_scheduler.prediction_type == "v_prediction":
                target = self.tr_noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.tr_noise_scheduler.prediction_type}"
                )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs["loss_total"] = loss
        return outputs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        nframes,
        height,
        width,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            nframes,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], dtype=self.dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(self.device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=self.device, dtype=self.dtype
                )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device=self.device, dtype=self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def forward_sample_multicontrol(self, inputs, outputs):
        gt_image_seq = inputs["label_img_seq"]
        b, c, t, h, w = gt_image_seq.size()
        ref_image = inputs["reference_img"]
        do_classifier_free_guidance = self.guidance_scale > 1.0

        if self.args.combine_clip_local:
            if self.args.drop_ref > 0:
                refer_latents = self.clip_encode_image_local_uncond(
                    ref_image,
                    self.args.num_inf_images_per_prompt,
                    do_classifier_free_guidance,
                )
            else:
                refer_latents = self.clip_encode_image_local(
                    ref_image,
                    self.args.num_inf_images_per_prompt,
                    do_classifier_free_guidance,
                )
        else:
            refer_latents = self.clip_encode_image_global(
                ref_image,
                self.args.num_inf_images_per_prompt,
                do_classifier_free_guidance,
            )

        if self.args.ref_null_caption:  # test must use null caption
            text = inputs["input_text"]
            text = ["" for i in text]
            text_embeddings = self.text_encode(
                text,
                num_images_per_prompt=self.args.num_inf_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=None,
            )

        controlnet_cond_list = []
        controlnet_conditioning_scale_list = []
        # Prepare conditioning image
        controlnet_conditioning_scale = self.controlnet_conditioning_scale
        controlnet_conditioning_scale_cond = self.controlnet_conditioning_scale_cond
        controlnet_conditioning_scale_ref = self.controlnet_conditioning_scale_ref
        image_pose = self.prepare_image(
            image=inputs["cond_img_seq"].to(dtype=self.dtype),
            width=w,
            height=h,
            batch_size=b * self.args.num_inf_images_per_prompt,
            num_images_per_prompt=self.args.num_inf_images_per_prompt,
            device=self.device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        controlnet_cond_list.append(image_pose)
        controlnet_conditioning_scale_list.append(controlnet_conditioning_scale_cond)

        # Prepare ref image
        if self.args.refer_sdvae:
            reference_latents_controlnet = self.image_encoder(
                inputs["reference_img_controlnet"]
            )
            reference_latents_controlnet = reference_latents_controlnet.to(
                dtype=self.dtype
            )
        else:
            reference_latents_controlnet = inputs["reference_img_controlnet"].to(
                dtype=self.dtype
            )

        reference_latents_controlnet = self.prepare_image(
            image=reference_latents_controlnet,
            width=w,
            height=h,
            batch_size=b * self.args.num_inf_images_per_prompt,
            num_images_per_prompt=self.args.num_inf_images_per_prompt,
            device=self.device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        controlnet_cond_list.append(reference_latents_controlnet)
        controlnet_conditioning_scale_list.append(controlnet_conditioning_scale_ref)

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(
            self.args.num_inference_steps, device=self.device
        )
        timesteps = self.noise_scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        gen_height = h
        gen_width = w
        generator = inputs["generator"]

        latents = self.prepare_latents(
            b * self.args.num_inf_images_per_prompt,
            num_channels_latents,
            self.args.nframes,
            gen_height,
            gen_width,
            generator,
            latents=None,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        # Denoising loop
        num_warmup_steps = (
            len(timesteps) - self.args.num_inference_steps * self.noise_scheduler.order
        )
        with self.progress_bar(total=self.args.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.noise_scheduler.scale_model_input(
                    latent_model_input, t
                )

                # controlnet(s) inference
                if (
                    self.args.ref_null_caption
                ):  # null caption input for ref controlnet path
                    raise NotImplementedError
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=[
                            refer_latents,
                            text_embeddings,
                            refer_latents,
                        ],
                        controlnet_cond=[
                            image_pose,
                            reference_latents_controlnet,
                            depth_image,
                        ],
                        conditioning_scale=[
                            controlnet_conditioning_scale_cond,
                            controlnet_conditioning_scale_ref,
                            controlnet_conditioning_scale,
                        ],
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=[refer_latents] * len(self.args.conds),
                        controlnet_cond=controlnet_cond_list,
                        conditioning_scale=controlnet_conditioning_scale_list,
                        nframes=self.args.nframes,
                        return_dict=False,
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=refer_latents,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample.to(dtype=self.dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Post-processing
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        gen_img = self.image_decoder(latents)
        gen_img = rearrange(gen_img, "(b t) c h w -> b c t h w", t=self.args.nframes)

        outputs["logits_img_seq"] = gen_img
        return outputs


def inner_collect_fn(
    args, inputs, outputs, log_dir, global_step, eval_save_filename="eval_visu"
):
    visu_save = getattr(args, "visu_save", False)

    rank = get_rank()
    if rank == -1:
        splice = ""
    else:
        splice = "_" + str(rank)
    if global_step <= 0:
        eval_log_dir = os.path.join(log_dir, eval_save_filename)
    else:
        eval_log_dir = os.path.join(log_dir, "eval_step_%d" % (global_step))
    ensure_dirname(eval_log_dir)

    if not visu_save:  # for every different combination, we build a new dir to save
        gt_save_path = os.path.join(eval_log_dir, "gt")
        ensure_dirname(gt_save_path)
        pred_save_path = os.path.join(
            eval_log_dir,
            f"pred_gs{args.guidance_scale}_scale-cond{args.controlnet_conditioning_scale_cond}-ref{args.controlnet_conditioning_scale_ref}",
        )
        ensure_dirname(pred_save_path)
        cond_save_path = os.path.join(eval_log_dir, "cond")
        ensure_dirname(cond_save_path)
        ref_save_path = os.path.join(eval_log_dir, "ref")
        ensure_dirname(ref_save_path)
        ref_control_save_path = os.path.join(eval_log_dir, "ref_control")
        ensure_dirname(ref_control_save_path)

    if rank in [-1, 0]:
        logger.warning(eval_log_dir)

        # Save Model Setting
        type_output = [
            int,
            float,
            str,
            bool,
            tuple,
            dict,
            type(None),
        ]
        setting_output = {
            item: getattr(args, item)
            for item in dir(args)
            if type(getattr(args, item)) in type_output and not item.startswith("__")
        }
        data2file(setting_output, os.path.join(eval_log_dir, "Model_Setting.json"))

    dl = {**inputs, **{k: v for k, v in outputs.items() if k.split("_")[0] == "logits"}}
    ld = dl2ld(dl)

    # l = ld[0]["logits_imgs"].shape[0]

    for _, sample in enumerate(ld):
        if visu_save:
            save_filename = sample["save_filename"]
            eval_log_dir_filename = os.path.join(eval_log_dir, save_filename)

            gt_save_path = os.path.join(eval_log_dir_filename, "gt")
            ensure_dirname(gt_save_path)
            pred_save_path = os.path.join(
                eval_log_dir_filename,
                f"pred_gs{args.guidance_scale}_scale-cond{args.controlnet_conditioning_scale_cond}-ref{args.controlnet_conditioning_scale_ref}",
            )
            ensure_dirname(pred_save_path)
            cond_save_path = os.path.join(eval_log_dir_filename, "cond")
            ensure_dirname(cond_save_path)
            ref_save_path = os.path.join(eval_log_dir_filename, "ref")
            ensure_dirname(ref_save_path)
            ref_control_save_path = os.path.join(eval_log_dir_filename, "ref_control")
            ensure_dirname(ref_control_save_path)

        _name = "nuwa"
        if "input_text" in sample:
            _name = sample["input_text"][:200]
        if "img_key" in sample:
            _name = sample["img_key"]
        elif "img_key_seq" in sample:
            sample["img_key_seq"] = sample["img_key_seq"].split(";")
            _name = sample["img_key_seq"][0]
        char_remov = [os.sep, ".", "/"]
        for c in char_remov:
            _name = _name.replace(c, "")
        prefix = "_".join(_name.split(" "))
        # prefix += splice
        # postfix = '_' + str(round(time.time() * 10))
        postfix = ""

        try:
            ref_img_size = sample["reference_img_size"]
        except:
            ref_img_size = None

        if "label_imgs" in sample:
            image = sample["label_imgs"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(
                image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
            )[0]
            try:
                image.save(os.path.join(gt_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving label_imgs: {e}")
        if "label_img_seq" in sample:
            os.makedirs(os.path.join(gt_save_path, prefix + postfix), exist_ok=True)
            for i in range(sample["label_img_seq"].shape[1]):
                image = sample["label_img_seq"][:, i]
                image = (image / 2 + 0.5).clamp(0, 1)
                image = tensor2pil(
                    image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
                )[0]
                try:
                    image.save(
                        os.path.join(
                            gt_save_path, prefix + postfix, sample["img_key_seq"][i]
                        )
                    )
                except Exception:
                    try:
                        image.save(
                            os.path.join(
                                gt_save_path,
                                prefix + postfix,
                                sample["img_key_seq"][i] + ".png",
                            )
                        )
                    except Exception as e:
                        print(
                            f"some errors happened in saving label_img_seq {os.path.join(gt_save_path, prefix + postfix, sample['img_key_seq'][i])}: {e}"
                        )

        if "logits_imgs" in sample:
            image = tensor2pil(
                sample["logits_imgs"],
                resize_img=args.pos_resize_img,
                img_target_size=ref_img_size,
            )[0]
            try:
                image.save(os.path.join(pred_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving logits_imgs: {e}")
        if "logits_img_seq" in sample:
            os.makedirs(os.path.join(pred_save_path, prefix + postfix), exist_ok=True)
            for i in range(sample["logits_img_seq"].shape[1]):
                image = sample["logits_img_seq"][:, i]
                image = tensor2pil(
                    image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
                )[0]
                try:
                    image.save(
                        os.path.join(
                            pred_save_path, prefix + postfix, sample["img_key_seq"][i]
                        )
                    )
                except Exception:
                    try:
                        image.save(
                            os.path.join(
                                pred_save_path,
                                prefix + postfix,
                                sample["img_key_seq"][i] + ".png",
                            )
                        )
                    except Exception as e:
                        print(f"some errors happened in saving logits_imgs: {e}")
        if "cond_imgs" in sample:  # pose
            image = tensor2pil(
                sample["cond_imgs"],
                resize_img=args.pos_resize_img,
                img_target_size=ref_img_size,
            )[0]
            try:
                image.save(os.path.join(cond_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving cond_imgs: {e}")
        if "cond_img_seq" in sample:
            os.makedirs(os.path.join(cond_save_path, prefix + postfix), exist_ok=True)
            for i in range(sample["cond_img_seq"].shape[1]):
                image = sample["cond_img_seq"][:, i]
                image = tensor2pil(
                    image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
                )[0]
                try:
                    image.save(
                        os.path.join(
                            cond_save_path, prefix + postfix, sample["img_key_seq"][i]
                        )
                    )
                except Exception:
                    try:
                        image.save(
                            os.path.join(
                                cond_save_path,
                                prefix + postfix,
                                sample["img_key_seq"][i] + ".png",
                            )
                        )
                    except Exception as e:
                        print(f"some errors happened in saving cond_img_seq: {e}")
        if "reference_img" in sample:
            image = sample["reference_img"]
            image = (
                image
                * torch.tensor(
                    [0.26862954, 0.26130258, 0.27577711], device=image.device
                ).view(3, 1, 1)
                + torch.tensor(
                    [0.48145466, 0.4578275, 0.40821073], device=image.device
                ).view(3, 1, 1)
            ).clamp(0, 1)
            image = tensor2pil(
                image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
            )[0]
            try:
                image.save(os.path.join(ref_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving reference_imgs: {e}")
        if "reference_img_controlnet" in sample:
            image = sample["reference_img_controlnet"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(
                image, resize_img=args.pos_resize_img, img_target_size=ref_img_size
            )[0]
            try:
                image.save(
                    os.path.join(ref_control_save_path, prefix + postfix + ".png")
                )
            except Exception as e:
                print(f"some errors happened in saving reference_img_controlnet: {e}")
    return gt_save_path, pred_save_path


def tensor2pil(images, resize_img=False, img_target_size=None):
    # c, h, w
    images = images.cpu().permute(1, 2, 0).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        if resize_img:
            assert img_target_size is not None
            img_target_size = img_target_size.squeeze()
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L").resize(img_target_size)
                for image in images
            ]
        else:
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
    else:
        if resize_img:
            assert img_target_size is not None
            img_target_size = img_target_size.squeeze()
            pil_images = [
                Image.fromarray(image).resize(img_target_size) for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

    return pil_images
