from utils.lib import *
from utils.dist import (
    is_main_process, get_world_size,
    reduce_dict, get_local_rank, synchronize,
    get_rank,)
from utils.metric_logger import log_dict_to_wandb, setup_wandb, log_img_to_wandb
from utils.misc import humanbytes
from utils.deepspeed import get_deepspeed_config, fp32_to_fp16
import deepspeed
from torch import nn
import torch.nn.functional as F
from utils.basic_utils import move_to_cuda
from utils.common import ensure_directory
from utils.wutils_ldm import (
    complex_to_device, logger, ensure_dirname, file2data, data2file,
    Meter, Timer, adaptively_load_state_dict, get_parameters, ldm_tensor2img_wt, ldm_tensor2img)
import wandb


class WarmupLinearLR(T.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio*max_iter)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step > tot_step:
            step = tot_step
        return max(0, (tot_step-step)/(tot_step-warmup_step))

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [
            max(self.min_lr, base_lr * warmup_factor)
            for base_lr in self.base_lrs
        ]


class WarmupLinearConstantLR(T.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio*max_iter)
        super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step >= warmup_step:
            return 1. # constant base lr

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [
            max(self.min_lr, base_lr * warmup_factor)
            for base_lr in self.base_lrs
        ]


class Agent():
    def __init__(self, args, model=None, optimizer=None, scheduler=None):
        super().__init__()
        self.args, self.model = args, model
        self.log_dir = args.log_dir
        if optimizer is None:
            if self.args.do_train:
                logger.warning("Missing optimizer, check if its intended to train without optimizer")
        self.optimizer = optimizer
        if scheduler is None:
            if self.args.do_train:
                logger.warning("Missing scheduler, check if its intended to train without scheduler")
        self.scheduler = scheduler

        self.pretrained_model = self.args.pretrained_model
        # Load Pretrained Models.
        if self.pretrained_model:
            if not self.args.deepspeed:
                self.from_pretrained(self.pretrained_model)
            # else:
            #     if not os.path.exists(self.pretrained_model + "/latest"):
            #         logger.warning(
            #             f"Pretrained deepspeed checkpoint path does not exists, {self.pretrained_model + '/latest'}")
        self.scaler = T.cuda.amp.GradScaler()
        self.local_rank = get_local_rank()
        self.rank = get_rank()
        
        self.enable_collect = True
        if is_main_process:
            ensure_directory(self.log_dir)
        self.metric_filename = os.path.join(self.log_dir, 'metric.json')
        self.last_checkpoint_filename = 'last.pth'
        # self.best_checkpoint_filename = 'best.pth'
        # self.each_checkpoint_filename = 'epoch%s.pth'
        self.epoch = -1
        self.global_step = -1

        if self.args.n_gpu == 0:
            logger.warning("No support on CPU")
        self.device = T.device("cuda")

        self.debug_dataloader = getattr(args, 'debug_dataloader', False)

    def setup_wandb(self):
        if WANDB_ENABLE and not self.args.debug:
            setup_wandb(self.args, project=self.args.wandb_project,
                        name=self.args.project_name)

    def log_dict_to_wandb(self, log_dict, step=-1):
        if WANDB_ENABLE and not self.args.debug:
            if step == -1:
                step = self.global_step
            log_dict_to_wandb(log_dict, step)

    def log_img_to_wandb(self, label_imgs, cond_imgs, ref_imgs, pred_imgs, step=-1, prefix=''):
        if WANDB_ENABLE and not self.args.debug:
            if step == -1:
                step = self.global_step
            img_dict = defaultdict(list)

            for i in range(len(pred_imgs)):
                img_dict[f"{prefix}_pred_img"].append(wandb.Image(ldm_tensor2img(pred_imgs[i])))
                if label_imgs is not None:
                    img_dict[f"{prefix}_label_img"].append(wandb.Image(ldm_tensor2img(label_imgs[i], preprocess=True)))
                if cond_imgs is not None:
                    img_dict[f"{prefix}_cond_img"].append(wandb.Image(ldm_tensor2img(cond_imgs[i])))
                if ref_imgs is not None:
                    img_dict[f"{prefix}_ref_img"].append(wandb.Image(ldm_tensor2img(ref_imgs[i], preprocess=True)))
            log_img_to_wandb(img_dict, step)

    def update_metric_file(self, metric):
        if os.path.exists(self.metric_filename):
            r = file2data(self.metric_filename, printable=False)
            data2file(dict(r, **metric), self.metric_filename, override=True)
        else:
            data2file(metric, self.metric_filename)

    def reduce_dict(self, data):
        return reduce_dict(data)

    def reduce_mean(self, v):
        world_size = get_world_size()
        if world_size < 2:
            return v
        else:
            v = T.tensor(v).cuda()
            DIST.all_reduce(v)
            v = v.item() / world_size
        return v
    
    def move_model_to_cuda(self):
        if self.debug_dataloader: # debug only, ugly workaround to run on local small gpu servers
            return
        self.model.to(self.device)
        if self.optimizer is not None:
            if isinstance(self.optimizer, list):
                for i in range(len(self.optimizer)):
                    self.optimizer[i].load_state_dict(
                        complex_to_device(self.optimizer[i].state_dict(), device=self.device))
            else:
                self.optimizer.load_state_dict(
                    complex_to_device(self.optimizer.state_dict(), device=self.device))

    def prepare_dist_model(self):
        if not self.args.dist:
            logger.info('Successfully built models with %s parameters' % get_parameters(self.model))
            return
        if self.args.deepspeed:
            if isinstance(self.optimizer, list):
                raise ValueError("A list of optimizers are not supported with deepspeed, or IDK how to make it work with deepspeed")
            config = get_deepspeed_config(self.args)

            if self.pretrained_model and not self.pretrained_model.endswith(".pth"):
                print('Specify the load model path, not use deepspeed but the pytorch original load func')
                self.load_checkpoint_for_deepspeed_diff_gpu(self.pretrained_model) # load pt model with default pytorch

            self.model, self.optimizer, _, _ = deepspeed.initialize(
                config_params=config,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler)

            if self.pretrained_model and self.pretrained_model.endswith(".pth"):
                logger.warning(f'Loading pre-trained model from {self.pretrained_model}')
                tag = self.pretrained_model.split("/")[-1]
                dir_ = self.pretrained_model.replace(tag, "")
                self.model.load_checkpoint(
                    dir_, tag=tag,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                    load_module_only=True,
                    load_module_strict=False)

            if self.args.resume:
                if os.path.exists(os.path.join(self.log_dir, "latest")):
                    logger.warning(f'Resuming.... from {os.path.join(self.log_dir, "latest")}')
                    self.model.load_checkpoint(
                        self.log_dir,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True,
                        load_module_strict=True)
                    self.global_step = self.model.global_steps * self.args.gradient_accumulate_steps
                else:
                    logger.warning(f'Resuming failed, path does not exists {os.path.join(self.log_dir, "latest")}')
        else:
            self.model = T.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.args.find_unused_parameters)
        if not self.args.do_train:
            self.model.eval()
        logger.info('Successfully built models with %s parameters' % get_parameters(self.model))

    def prepare_batch(self, batch):
        batch = move_to_cuda(batch)
        if self.args.deepspeed:
            batch = fp32_to_fp16(batch)
        return batch

    def forward_step(self, batch):
        if self.args.use_amp:
            with T.autocast(device_type='cuda'):
                out = self.model(batch)
        else:
            out = self.model(batch)
        return out

    def backward_step(self, loss):
        if self.args.deepspeed:
            self.model.backward(loss)
        elif self.args.use_amp:
            self.scaler.scale(loss).backward()
            
        else:
            loss.backward()
            
    def grad_clip(self):
        if self.args.deepspeed:
            # haddled by deepspeed
            return
        if self.args.max_grad_norm > 0:
            if self.args.use_amp:
                    self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned
                # params are unscaled, clips as usual:
            T.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm)

    def step(self, optimizer_idx=-1):
        if optimizer_idx >= 0 and isinstance(self.optimizer, list):
            optimizer = self.optimizer[optimizer_idx]
        else:
            optimizer = self.optimizer
        if self.args.deepspeed:
            # Not sure how to step one optimizer at a time
            self.model.step()
        elif self.args.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

    def train(self, train_loader, eval_loader=None, use_tqdm=None,
              inner_collect_fn=None):
        if self.args.resume:
            if not self.args.deepspeed:
                if os.path.exists(os.path.join(self.log_dir, self.last_checkpoint_filename)):
                    self.load_checkpoint(self.last_checkpoint_filename)
        else:
            if is_main_process():
                logger.warning('Dangerous! You set resume=False. Auto cleaning all the logs under %s' % self.log_dir)
                ensure_dirname(self.log_dir, override=True)
        
        self.move_model_to_cuda()
        if self.args.dist:
            self.prepare_dist_model()

        epoch_iter = range(self.epoch + 1, self.args.epochs, 1)
        if len(epoch_iter):
            logger.warning('Start train & val phase...')
        else:
            logger.warning('Skip train & val phase...')
        logger.warning(
            f'Train examples: {len(train_loader.dataset)}, image size {train_loader.dataset.img_size},\n'
            f'\t\tVal examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n'
            f'\t\tepochs: {self.args.epochs}, iters: {self.args.num_iters}, \n'
            f'\t\titer_per_ep: {self.args.iter_per_ep}, eval_step: {self.args.eval_step}, save_step: {self.args.save_step},\n'
            f'\t\tglobal_batch_size: {self.args.train_batch_size}, local_batch_size: {self.args.local_train_batch_size}.')

        # Train & Eval phase
        for epoch in epoch_iter:
            self.epoch = epoch
            # Train phase
            train_meter, train_time = self.train_fn(train_loader,
                                                    use_tqdm=use_tqdm)
            logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, epoch + 1, self.args.epochs, train_time, train_meter.avg))
            if not isinstance(train_meter.avg, dict):
                raise ValueError(type(train_meter.avg))
            metric = {'Epoch%s' % (epoch + 1): {'train': {**train_meter.avg, **{'time': train_time}}}}

            if is_main_process():
                self.update_metric_file(metric)
            if (epoch + 1) % self.args.save_step == 0:
                self.save_checkpoint(self.last_checkpoint_filename)

            if (epoch + 1) % self.args.eval_step == 0:
                if eval_loader:
                    eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                         use_tqdm=use_tqdm)
                    logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                                (self.rank, epoch + 1, self.args.epochs, eval_time, eval_meter.avg))

                    # Update metric with eval metrics
                    metric['Epoch%s' % (epoch + 1)].update({'eval': {**eval_meter.avg, **{'time': eval_time}}})

                    # Save metric file
                    if is_main_process():
                        self.update_metric_file(metric)

    def train_fn(self, train_loader, use_tqdm=True):
        self.model.train()
        train_meter = Meter()
        train_timer = Timer()
        train_iter = tqdm(train_loader, total=len(train_loader), disable=not use_tqdm)
        for step, inputs in enumerate(train_iter):
            for optimizer_idx in range(len(self.optimizer)):
                if not getattr(self.optimizer[optimizer_idx], 'is_enabled', lambda x: True)(self.epoch):
                    continue

                inputs['epoch'] = self.epoch
                inputs['global_step'] = self.epoch * len(train_loader) + step
                self.global_step = inputs['global_step'] 
                inputs['optimizer_idx'] = optimizer_idx

                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)
                self.check_outputs(outputs)

                if optimizer_idx == 0:
                    self.backward_step(outputs['loss_total'])
                else:
                    self.backward_step(outputs[f'loss_total_{optimizer_idx}'])
                    # outputs['loss_total_%s' % optimizer_idx].backward()

                if (step + 1) % self.args.gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                    self.grad_clip()
                    self.step(optimizer_idx)

                metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                train_meter.update(metric_and_loss)

            if self.scheduler:
                self.scheduler.step()

            train_iter.set_description("Metering:" + str(train_meter))
        train_time = train_timer.elapse(True)
        return train_meter, train_time

    def eval(self, eval_loader, inner_collect_fn=None, use_tqdm=True, enc_dec_only=False):
        # This function is used to do evaluating after training.
        if not self.pretrained_model:
            logger.warning('You should create a new config file and specify pretrained_model in Args when using eval.')
        # Wrap models before evaluating. This will support ddp evaluating.
        self.move_model_to_cuda()
        if self.args.dist:
            self.prepare_dist_model()
        eval_meter, eval_time = self.eval_fn(
            eval_loader, inner_collect_fn=inner_collect_fn,
            use_tqdm=use_tqdm, enc_dec_only=enc_dec_only)
        logger.info('[Rank %s] Valid Time: %s\n %s' % (self.rank, eval_time, eval_meter.avg))

    def eval_trainsample(self, train_loader, eval_loader, inner_collect_fn=None, use_tqdm=True, enc_dec_only=False):
        # This function is used to do evaluating with training sample.
        if not self.pretrained_model:
            logger.warning('You should create a new config file and specify pretrained_model in Args when using eval.')
        # Wrap models before evaluating. This will support ddp evaluating.
        self.move_model_to_cuda()
        if self.args.dist:
            self.prepare_dist_model()
        eval_meter, eval_time = self.eval_fn_trainsample(
            train_loader, eval_loader, inner_collect_fn=inner_collect_fn,
            use_tqdm=use_tqdm, enc_dec_only=enc_dec_only)
        logger.info('[Rank %s] Valid Time: %s\n %s' % (self.rank, eval_time, eval_meter.avg))

    def eval_demo_pre(self):
        if not self.pretrained_model:
            logger.warning('You should create a new config file and specify pretrained_model in Args when using eval.')
        # Wrap models before evaluating. This will support ddp evaluating.
        self.move_model_to_cuda()
        if self.args.dist:
            self.prepare_dist_model()

    def eval_demo_run_masked(self, input_batch, eval_dataset, enc_dec_only=False):
        input_batch = eval_dataset.preprocess_masked_input(*input_batch)
        output_image = self.eval_fn_demo(
            input_batch, enc_dec_only=enc_dec_only)
        return output_image

    def eval_fn(
        self,
        eval_loader,
        inner_collect_fn=None,
        use_tqdm=True,
        compute_fid=True,
        enc_dec_only=False,
        train_eval_input=None,
    ):
        # TODO Note that eval_fn supports ddp. So we do not need to unwrap things here.
        self.model.eval()
        eval_meter = Meter()
        eval_timer = Timer()
        eval_save_filename = self.args.eval_save_filename
        if enc_dec_only:
            eval_save_filename += "_enc_dec_only"
        with T.no_grad():
            eval_loader = (
                tqdm(eval_loader, total=len(eval_loader)) if use_tqdm else eval_loader
            )
            for batch_idx, inputs in enumerate(eval_loader):
                T.cuda.empty_cache()
                if enc_dec_only:
                    inputs["enc_dec_only"] = True
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)
                metric_and_loss = {
                    k: v
                    for k, v in outputs.items()
                    if k.split("_")[0] in ["metric", "loss"]
                }

                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                if inner_collect_fn and self.enable_collect:
                    remove_key = inputs.pop("enc_dec_only", None)
                    gt_save_path, pred_save_path = inner_collect_fn(
                        self.args,
                        inputs,
                        outputs,
                        self.log_dir,
                        self.global_step,
                        eval_save_filename,
                    )

                if batch_idx == 0:
                    inputs = defaultdict(lambda: None, inputs)
                    try:
                        label_imgs = inputs["label_imgs"]
                        cond_imgs = inputs["cond_imgs"]
                        ref_imgs = inputs["reference_img"]
                        pred_imgs = outputs["logits_imgs"]
                    except:  # for video model
                        label_imgs = inputs["label_img_seq"][:, :, 0]
                        cond_imgs = inputs["cond_img_seq"][:, :, 0]
                        ref_imgs = inputs["reference_img"]
                        pred_imgs = outputs["logits_img_seq"][:, :, 0]
                    self.log_img_to_wandb(
                        label_imgs, cond_imgs, ref_imgs, pred_imgs, prefix="val"
                    )

            if (
                train_eval_input
            ):  # run a simple-round training sample to check if it is over-fitting
                print(
                    "run a single-round training sample inference to check if over-fitting"
                )
                inputs = train_eval_input
                T.cuda.empty_cache()
                if enc_dec_only:
                    inputs["enc_dec_only"] = True
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)

                ### vis image for training single round sample ###
                inputs = defaultdict(lambda: None, inputs)
                try:
                    label_imgs = inputs["label_imgs"]
                    cond_imgs = inputs["cond_imgs"]
                    ref_imgs = inputs["reference_img"]
                    pred_imgs = outputs["logits_imgs"]
                except:  # for video model
                    label_imgs = inputs["label_img_seq"][:, :, 0]
                    cond_imgs = inputs["cond_img_seq"][:, :, 0]
                    ref_imgs = inputs["reference_img"]
                    pred_imgs = outputs["logits_img_seq"][:, :, 0]
                self.log_img_to_wandb(
                    label_imgs, cond_imgs, ref_imgs, pred_imgs, prefix="train"
                )

        eval_meter = self.get_eval_metrics(
            eval_meter, eval_save_filename, gt_save_path, pred_save_path
        )
        eval_time = eval_timer.elapse(True)

        self.model.train()
        return eval_meter, eval_time

    def eval_fn_demo(self, input_batch, enc_dec_only=False):
        self.model.eval()
        with T.no_grad():
            inputs = input_batch
            T.cuda.empty_cache()
            if enc_dec_only:
                inputs['enc_dec_only'] = True
            inputs = self.prepare_batch(inputs)
            outputs = self.forward_step(inputs)

            from PIL import Image
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
                        pil_images = [Image.fromarray(image.squeeze(), mode="L").resize(img_target_size) for image in images]
                    else:
                        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
                else:
                    if resize_img:
                        assert img_target_size is not None
                        img_target_size = img_target_size.squeeze()
                        pil_images = [Image.fromarray(image).resize(img_target_size) for image in images]
                    else:
                        pil_images = [Image.fromarray(image) for image in images]

                return pil_images

            output_image = tensor2pil(outputs['logits_imgs'].squeeze(), resize_img=self.args.pos_resize_img, img_target_size=None)[0]
        return output_image


    def get_eval_metrics(self, eval_meter, eval_save_filename, gt_save_path=None, pred_save_path=None):
        synchronize()
        if self.global_step <= 0:
            eval_log_dir = os.path.join(self.log_dir, eval_save_filename)
        else:
            eval_log_dir = os.path.join(self.log_dir, 'eval_step_%d' % (self.global_step))

        if gt_save_path:
            gt_folder = gt_save_path
        else:
            gt_folder = os.path.join(eval_log_dir, 'gt_png')
            if not os.path.exists(gt_folder):
                gt_folder = os.path.join(eval_log_dir, 'gt')

        if pred_save_path:
            pred_folder = pred_save_path
        else:
            pred_folder = os.path.join(eval_log_dir, 'pred_png')
            if not os.path.exists(pred_folder):
                pred_folder = os.path.join(eval_log_dir, 'pred')

        if os.path.exists(gt_folder) and os.path.exists(pred_folder):
            if is_main_process():
                try: 
                    from tool.cleanfid.fid import compute_fid
                    result = compute_fid(gt_folder, pred_folder)
                    logger.info(f"FID is {result}")
                    eval_meter.update({'FID': result})
                except Exception as e:
                    logger.warning(f"Failed to calculate FID, {e}")
        else:
            logger.warning(
                f'Failed to calculate FID, gt {gt_folder}, {os.path.exists(gt_folder)}\npred {pred_folder}, {os.path.exists(pred_folder)}')
        
        gt_folder = os.path.join(eval_log_dir, 'gt_gif')
        if not os.path.exists(gt_folder):
            gt_folder = os.path.join(eval_log_dir, 'gt')
        pred_folder = os.path.join(eval_log_dir, 'pred_gif')
        if not os.path.exists(pred_folder):
            pred_folder = os.path.join(eval_log_dir, 'pred')

        if os.path.exists(gt_folder) and os.path.exists(pred_folder):
            if is_main_process():
                try: 
                    from tool.metrics.metric_center import get_all_eval_scores
                    result = get_all_eval_scores(
                        self.args.root_dir, gt_folder, pred_folder,
                        sample_duration=self.args.max_video_len,
                        metrics=['fid-img', 'fid-vid', 'fvd'])
                    logger.info(f"Video gen eval {result}")
                    eval_meter.update(result)
                except Exception as e:
                    logger.warning(f"Failed to eval video gen, {e}")
        else:
            logger.warning(
                f'Failed to eval video gen, gt {gt_folder}, {os.path.exists(gt_folder)}\npred {pred_folder}, {os.path.exists(pred_folder)}')

        if self.args.eval_visu and is_main_process() and len({**eval_meter.avg}):
            json.dump({**eval_meter.avg}, open(f"{eval_log_dir}/metrics.json", "w"))
        synchronize()
        return eval_meter
    
    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True, inner_collect_fn=None):

        if self.args.num_iters:
            logger.warning('Start train & val phase...')
            self.setup_wandb()
        else:
            logger.warning('Skip train & val phase...')
            return
        logger.warning(
            f'Train examples: {len(train_loader.dataset)}, Val examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n'
            f'\t\tepochs: {self.args.epochs}, iters: {self.args.num_iters}, \n'
            f'\t\titer_per_ep: {self.args.iter_per_ep}, eval_step: {self.args.eval_step}, save_step: {self.args.save_step},\n'
            f'\t\tglobal_batch_size: {self.args.train_batch_size}, local_batch_size: {self.args.local_train_batch_size}.')

        # Train & Eval phase
        train_pbar = tqdm(total=len(train_loader), disable=not use_tqdm)
        train_meter = Meter()
        train_timer = Timer()
        metric = defaultdict(dict)
        if self.global_step > 0:
            train_pbar.update(self.global_step)
        else:
            self.global_step = 0
        
        if self.args.eval_before_train and self.global_step == 0:
            logger.warning("Saving model...")
            self.save_checkpoint(str(self.global_step) + '.pth')
            logger.warning("Evaluating...")
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                    use_tqdm=use_tqdm)
                logger.info('[Rank %s] Valid  before train. Time: %s\n %s' %
                            (self.rank, eval_time, eval_meter.avg))
                # Update metric with eval metrics
                val_meter_log = {**eval_meter.avg}
                self.log_dict_to_wandb({f'{k}_ep': v for k, v in val_meter_log.items()}, step=0)
                val_meter_log.update({'time': eval_time})
                metric['Epoch 0'].update({'eval': val_meter_log})

                # Save metric file
                if is_main_process():
                    self.update_metric_file(metric)

        self.model.train()
        train_iter = iter(train_loader)
        while self.global_step < len(train_loader):
            try:
                inputs = next(train_iter)
            except StopIteration:
                logger.warning("Reaching end of the train_loader, terminating training loop")
                break
            self.epoch = (self.global_step + 1) // self.args.iter_per_ep
            for optimizer_idx in range(len(self.optimizer)):
                if not getattr(self.optimizer[optimizer_idx], 'is_enabled', lambda x: True)(self.epoch):
                    continue

                inputs['epoch'] = self.epoch
                inputs['global_step'] = self.global_step
                inputs['optimizer_idx'] = optimizer_idx

                inputs = self.prepare_batch(inputs)
                T.cuda.empty_cache()
                outputs = self.forward_step(inputs)

                self.check_outputs(outputs)
                T.cuda.empty_cache()
                if optimizer_idx == 0:
                    self.backward_step(outputs['loss_total'])
                else:
                    self.backward_step(outputs[f'loss_total_{optimizer_idx}'])

                if (self.global_step + 1) % self.args.gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                    self.grad_clip()
                    self.step(optimizer_idx)

                metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                train_meter.update(metric_and_loss)
                self.log_dict_to_wandb(metric_and_loss)

            if self.scheduler:
                self.scheduler.step()

            # if (self.global_step + 1) % (self.args.save_step*self.args.iter_per_ep) == 0:
            #     self.save_checkpoint(str(self.global_step) + '.pth')
            train_pbar.set_description("Metering:" + str(train_meter))
            train_time = train_timer.elapse(True)

            if (self.global_step + 1) % self.args.iter_per_ep == 0:
                logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, self.epoch + 1, self.args.epochs, train_time, train_meter.avg))
                if not isinstance(train_meter.avg, dict):
                    raise ValueError(type(train_meter.avg))
                tr_meter_log = {**train_meter.avg,}
                self.log_dict_to_wandb({f'{k}_ep': v for k, v in tr_meter_log.items()})
                tr_meter_log.update({'time': train_time})
                metric['Epoch%s' % (self.epoch + 1)].update( {'train': tr_meter_log})

                if is_main_process():
                    self.update_metric_file(metric)
                if (self.epoch + 1) % self.args.save_step == 0:
                    logger.warning("Saving model...")
                    self.save_checkpoint(str(self.global_step) + '.pth')
                    self.save_checkpoint(self.last_checkpoint_filename)
                        # copy_file(self.last_checkpoint_filename, self.each_checkpoint_filename % str(epoch + 1),
                        #           override=True)  # TODO sometimes we need to copy file

                if (self.epoch + 1) % self.args.eval_step == 0:
                    logger.warning("Evaluating...")
                    if eval_loader:
                        eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                            use_tqdm=use_tqdm)
                        logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                                    (self.rank, self.epoch + 1, self.args.epochs, eval_time, eval_meter.avg))
                        # Update metric with eval metrics
                        val_meter_log = {**eval_meter.avg}
                        self.log_dict_to_wandb({f'{k}_ep': v for k, v in val_meter_log.items()})
                        val_meter_log.update({'time': eval_time})
                        metric['Epoch%s' % (self.epoch + 1)].update({'eval': val_meter_log})

                        # Save metric file
                        if is_main_process():
                            self.update_metric_file(metric)
                train_meter = Meter()
                train_timer = Timer()
            self.global_step += 1
            train_pbar.update(1)
        if (self.epoch + 1) % self.args.save_step != 0:
            logger.warning("Saving model...")
            self.save_checkpoint(str(self.global_step) + '.pth')
            self.save_checkpoint(self.last_checkpoint_filename)

        if (self.epoch + 1) % self.args.eval_step != 0:
            logger.warning("Evaluating...")
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(
                    eval_loader, inner_collect_fn=inner_collect_fn,
                    use_tqdm=use_tqdm)
                logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                            (self.rank, self.epoch + 1, self.args.epochs, eval_time, eval_meter.avg))
                # Update metric with eval metrics
                val_meter_log = {**eval_meter.avg}
                self.log_dict_to_wandb({f'{k}_ep': v for k, v in val_meter_log.items()})
                val_meter_log.update({'time': eval_time})
                metric['Epoch%s' % (self.epoch + 1)].update({'eval': val_meter_log})

                # Save metric file
                if is_main_process():
                    self.update_metric_file(metric)

    def check_outputs(self, outputs):
        error_message = 'Model output must be a dict. The key must be "class_subclass" format.' \
                        ' "class" can only be loss, metric, or logits. "subclass" should be a string.' \
                        ' But got an unexpected key %s'
        loss_total_list = [e for e in outputs.keys() if e.startswith('loss_total')]
        if not loss_total_list:
            raise ValueError('Model output must contain a key startswith "loss_total"!')

        for k, v in outputs.items():
            split_res = k.split('_')
            if len(split_res) < 2:
                raise ValueError(error_message % k)
            if k.split('_')[0] not in ['loss', 'metric', 'logits']:
                raise ValueError(error_message % k)
    
    def from_pretrained(self, pretrained_model):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load pretrained models into wrapped models, ensure self.models is CPU.")
        if isinstance(pretrained_model, str):
            logger.warning('Loading Pretrained Model Path: %s...' % pretrained_model)
            pretrained_dict = file2data(pretrained_model, map_location='cpu')
            if 'models' in pretrained_dict:
                pretrained_dict = pretrained_dict['models']
            elif 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
        else:
            logger.warning('Loading Given Pretrained Dict...')
            pretrained_dict = pretrained_model
        adaptively_load_state_dict(self.model, pretrained_dict)

    def load_checkpoint(self, checkpoint_filename):
        if not self.args.deepspeed:
            if hasattr(self.model, "module"):
                raise ValueError("Please do not load checkpoint into wrapped models, ensure self.models is CPU.")
            checkpoint = file2data(checkpoint_filename, map_location='cpu')
            adaptively_load_state_dict(self.model, checkpoint['models'])
            if isinstance(self.optimizer, list):
                if len(self.optimizer) > 1:
                    for i, optimizer in enumerate(self.optimizer):
                        adaptively_load_state_dict(self.optimizer[i], checkpoint['optimizer'][i])

                elif len(self.optimizer) == 1:
                    adaptively_load_state_dict(self.optimizer[0], checkpoint['optimizer'])
            else:
                adaptively_load_state_dict(self.optimizer, checkpoint['optimizer'])
            if self.scheduler:
                adaptively_load_state_dict(self.scheduler, checkpoint['scheduler'])

            self.epoch = checkpoint['epoch'] - 1
            self.global_step = checkpoint['global_step'] - 1

            # IMPORTANT! The models will be wrapped automatically.
            logger.warning('Loaded checkpoint %s of epoch %s (global_step %s)' % (
                checkpoint_filename, checkpoint['epoch'],
                checkpoint['global_step']))
        else:
            self.model.load_checkpoint(self.log_dir, checkpoint_filename)
            logger.warning('Loaded checkpoint %s' % (checkpoint_filename))

    def load_checkpoint_for_deepspeed_diff_gpu(self, checkpoint_filename):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load checkpoint into wrapped models, ensure self.models is CPU.")
        checkpoint = file2data(checkpoint_filename, map_location='cpu')
        adaptively_load_state_dict(self.model, checkpoint['module'])

        # IMPORTANT! The models will be wrapped automatically.
        logger.warning('Loaded checkpoint %s of global_step %s' % (
            checkpoint_filename,
            checkpoint['global_steps']))


    def save_checkpoint(self, checkpoint_filename):
        if not self.args.deepspeed:
            if not is_main_process():
                return
            checkpoint_filename = os.path.join(self.log_dir, checkpoint_filename)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            if isinstance(self.optimizer, list):
                if len(self.optimizer) > 1:
                    optimizer_to_save = [optimizer.state_dict() for optimizer in self.optimizer]
                elif len(self.optimizer) == 1:
                    optimizer_to_save = self.optimizer[0].state_dict()
            else:
                optimizer_to_save = self.optimizer.state_dict()
            checkpoint = {
                'models': model_to_save.state_dict(),
                'optimizer': optimizer_to_save,
                'epoch': self.epoch + 1,
                'global_step': self.global_step + 1
            }
            if self.scheduler:
                checkpoint['scheduler'] = self.scheduler.state_dict()
            data2file(checkpoint, checkpoint_filename, override=True)
            logger.warning('Saved epoch %s (global_step %s) to %s.' % (
                checkpoint['epoch'], checkpoint['global_step'],
                checkpoint_filename))
            return
        else:
            if self.args.debug and checkpoint_filename != self.last_checkpoint_filename:
                logger.warning(
                    f"skip saving models with deepspeed to "
                    f"{self.log_dir}/{checkpoint_filename} in debug mode")
                return
            self.model.save_checkpoint(self.log_dir, tag=checkpoint_filename)
            return

    def log_memory(self, ep=-1, step=-1):
        if ep == -1 and step == -1:
            step = self.global_step
            step_str = f"global step: {step},"
        else:
            step_str = f"ep: {ep}, step: {step},"

        memory = humanbytes(T.cuda.max_memory_allocated())
        lr_base = f'{self.optimizer.param_groups[0]["lr"]:.2e}'
        lr_head = f'{self.optimizer.param_groups[2]["lr"]:.2e}'
        lr_xmodal = f'{self.optimizer.param_groups[4]["lr"]:.2e}'
        self.log_dict_to_wandb({'lr_base': float(lr_base)}, step)
        self.log_dict_to_wandb({'lr_head': float(lr_head)}, step)
        self.log_dict_to_wandb({'lr_xmodal': float(lr_xmodal)}, step)
        return f"{step_str} lr_base: {lr_base}, " +\
            f"lr_head: {lr_head}, lr_xmodal: {lr_xmodal}, max memory: {memory}"

    def build_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = ["fc_mtm", "fc"]
        cross_modal_names = ["cross_modal", "i2t", "t2i"]
        lr_mult_head = self.args.lr_mult_head
        lr_mult_cross_modal = self.args.lr_mult_cross_modal

        wd = self.args.decay
        lr = self.args.lr

        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_cross_modal,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_cross_modal,
            },
        ]
        optzr = T.optim.AdamW(
            optimizer_grouped_parameters, lr=lr,
            betas=(0.9, 0.98))
        return optzr
    
    def setup_model_for_training(self):
        if self.args.resume:
            if not self.args.deepspeed:
                if os.path.exists(os.path.join(self.log_dir, self.last_checkpoint_filename)):
                    logger.warning(f'Resuming.... from {self.last_checkpoint_filename}')
                    self.load_checkpoint(self.last_checkpoint_filename)
            else:        
                if isinstance(self.optimizer, list):
                    raise ValueError("self.optimizer is a list, which is incompatible with deepspeed")
        else:
            if is_main_process():
                logger.warning('Dangerous! You set resume=False. Auto cleaning all the logs under %s' % self.log_dir)
                ensure_dirname(self.log_dir, override=True)
        if hasattr(self.model, "init_ddpm"):
            # for LDM only
            logger.warning('Init ddpm and freeze cond_model/first_stage_vqe_model.')
            self.model.init_ddpm()
        self.move_model_to_cuda()
        self.prepare_dist_model()
            


class Agent_LDM(Agent):
    def __init__(self, args, model, optimizer=None, scheduler=None):
        super().__init__(args, model, optimizer, scheduler)
        ### add to use stage 1 pretrain attr model to initialize the weight
        if self.args.stage1_pretrain_path is not None:
            print('### Using stage 1 attribute pretrain model to initialize the weights ###')
            pretrain_model_state = file2data(self.args.stage1_pretrain_path, map_location='cpu')
            # pre-process to the weight name
            new_state = {}
            for k, v in pretrain_model_state['module'].items():
                if k.startswith('controlnet.'):
                    new_key = k[:11] + 'nets.1.' + k[11:]
                    new_state[new_key] = v
                else:
                    new_state[k] = v
            ### load
            adaptively_load_state_dict(model, new_state)


    def train(self, train_loader, eval_loader=None, use_tqdm=True,
              inner_collect_fn=None):
        if self.args.resume:
            if not self.args.deepspeed:
                if os.path.exists(os.path.join(self.log_dir, self.last_checkpoint_filename)):
                    logger.warning(f'Resuming.... from {self.last_checkpoint_filename}')
                    self.load_checkpoint(self.last_checkpoint_filename)
        else:
            if is_main_process():
                logger.warning('Dangerous! You set resume=False. Auto cleaning all the logs under %s' % self.log_dir)
                ensure_dirname(self.log_dir, override=True)
        
        self.model.init_ddpm()
        self.move_model_to_cuda()
        if self.args.dist:
            self.prepare_dist_model()

        epoch_iter = range(self.epoch + 1, self.args.epochs, 1)
        if len(epoch_iter):
            logger.warning('Start train & val phase...')
        else:
            logger.warning('Skip train & val phase...')
        logger.warning(
            f'Train examples: {len(train_loader.dataset)}, image size {train_loader.dataset.img_size},\n'
            f'\t\tVal examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n'
            f'\t\tepochs: {self.args.epochs}, iters: {self.args.num_iters}, \n'
            f'\t\titer_per_ep: {self.args.iter_per_ep}, eval_step: {self.args.eval_step}, save_step: {self.args.save_step},\n'
            f'\t\tglobal_batch_size: {self.args.train_batch_size}, local_batch_size: {self.args.local_train_batch_size}.')

        # Train & Eval phase
        for epoch in epoch_iter:
            self.epoch = epoch
            # Train phase
            train_meter, train_time = self.train_fn(train_loader,
                                                    use_tqdm=use_tqdm)
            logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, epoch + 1, self.args.epochs, train_time, train_meter.avg))
            if not isinstance(train_meter.avg, dict):
                raise ValueError(type(train_meter.avg))
            metric = {'Epoch%s' % (epoch + 1): {'train': {**train_meter.avg, **{'time': train_time}}}}

            if is_main_process():
                self.update_metric_file(metric)
            if (epoch + 1) % self.args.save_step == 0:
                self.save_checkpoint(self.last_checkpoint_filename)

            if (epoch + 1) % self.args.eval_step == 0:
                if eval_loader:
                    eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                         use_tqdm=use_tqdm)
                    logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                                (self.rank, epoch + 1, self.args.epochs, eval_time, eval_meter.avg))
                    # Update metric with eval metrics
                    metric['Epoch%s' % (self.epoch + 1)].update({'eval': {**eval_meter.avg, **{'time': eval_time}}})

                    # Save metric file
                    if is_main_process():
                        self.update_metric_file(metric)

    def train_fn(self, train_loader, use_tqdm=True):
        self.model.train()
        train_meter = Meter()
        train_timer = Timer()
        train_iter = tqdm(train_loader, total=len(train_loader), disable=not use_tqdm)
        for step, inputs in enumerate(train_iter):
            if not getattr(self.optimizer, 'is_enabled', lambda x: True)(self.epoch * len(train_loader) + step):
                continue # adjust to sdm KL-VAE
            inputs = complex_to_device(inputs, self.device)

            inputs['epoch'] = self.epoch
            inputs['global_step'] = self.epoch * len(train_loader) + step
            self.global_step = inputs['global_step']

            inputs = self.prepare_batch(inputs)
            outputs = self.forward_step(inputs)

            self.check_outputs(outputs)
            self.backward_step(outputs['loss_total'])

            if not self.args.deepspeed:
                if (step + 1) % self.args.gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                    self.grad_clip()
                    self.step()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                self.grad_clip()
                self.step()

            metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}

            for k, v in metric_and_loss.items():
                metric_and_loss[k] = self.reduce_mean(v)
            train_meter.update(metric_and_loss)

            if (self.global_step + 1) % (int(getattr(self.args, 'save_setp', 8000)) + 1) == 0:
                self.save_checkpoint(str(self.global_step) + '.pth')
            train_iter.set_description("Metering:" + str(train_meter))
        train_time = train_timer.elapse(True)
        return train_meter, train_time

    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True, inner_collect_fn=None):
        if self.args.num_iters:
            logger.warning('Start train & val phase...')
            self.setup_wandb()
        else:
            logger.warning('Skip train & val phase...')
            return
        logger.warning(
            f'Train examples: {len(train_loader.dataset)}, image size {train_loader.dataset.img_size},\n'
            f'\t\tVal examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n'
            f'\t\tepochs: {self.args.epochs}, iters: {self.args.num_iters}, \n'
            f'\t\titer_per_ep: {self.args.iter_per_ep}, eval_step: {self.args.eval_step}, save_step: {self.args.save_step},\n'
            f'\t\tglobal_batch_size: {self.args.train_batch_size}, local_batch_size: {self.args.local_train_batch_size}.')

        # Train & Eval phase
        train_pbar = tqdm(total=len(train_loader), disable=not use_tqdm)
        train_meter = Meter()
        train_timer = Timer()
        metric = defaultdict(dict)
        if self.global_step > 0:
            train_pbar.update(self.global_step)
        else:
            self.global_step = 0

        if self.args.eval_before_train and self.global_step == 0:
            # logger.warning("Saving model...")
            # self.save_checkpoint(str(self.global_step) + '.pth')
            if eval_loader:
                if self.args.eval_enc_dec_only:
                    logger.warning("Evaluating enc_dec_only...")
                    eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                        use_tqdm=use_tqdm, enc_dec_only=True)
                    logger.info('[Rank %s] Valid enc_dec_only before train. Time: %s\n %s' %
                                (self.rank, eval_time, eval_meter.avg))
                    # Update metric with eval metrics
                    val_meter_log = {**eval_meter.avg}
                    self.log_dict_to_wandb({f'{k}_enc_dec_only': v for k, v in val_meter_log.items()}, step=0)
                    val_meter_log.update({'time': eval_time})
                    metric['Step 0_enc_dec_only'].update({'eval': val_meter_log})
                logger.warning("Evaluating ZS...")

                eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                    use_tqdm=use_tqdm)
                logger.info('[Rank %s] Valid  before train. Time: %s\n %s' %
                            (self.rank, eval_time, eval_meter.avg))
                # Update metric with eval metrics
                val_meter_log = {**eval_meter.avg}
                self.log_dict_to_wandb({f'{k}_step': v for k, v in val_meter_log.items()}, step=0)
                val_meter_log.update({'time': eval_time})
                metric['Step 0'].update({'eval': val_meter_log})

                # Save metric file
                if is_main_process():
                    self.update_metric_file(metric)
        # update tqdm to have previous info, load train_loader
        self.model.train()
        train_iter = iter(train_loader)
        while self.global_step < len(train_loader):
            try:
                inputs = next(train_iter)
            except StopIteration:
                logger.warning("Reaching end of the train_loader, terminating training loop")
                break

            self.epoch = (self.global_step + 1) // self.args.iter_per_ep
            if not getattr(self.optimizer, 'is_enabled', lambda x: True)(self.global_step):
                continue # adjust to sdm KL-VAE

            inputs['epoch'] = self.epoch
            inputs['global_step'] = self.global_step

            inputs = self.prepare_batch(inputs)
            T.cuda.empty_cache()
            outputs = self.forward_step(inputs)

            self.check_outputs(outputs)
            T.cuda.empty_cache()
            self.backward_step(outputs['loss_total'])

            if not self.args.deepspeed:
                if (self.global_step + 1) % self.args.gradient_accumulate_steps == 0 and outputs.get('logits_last', True):
                    self.grad_clip()
                    self.step()
                if self.scheduler:
                    self.scheduler.step()
            else:
                # gradient_accumulation handled by deepspeed
                self.grad_clip()
                self.step()

            metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
            for k, v in metric_and_loss.items():
                metric_and_loss[k] = self.reduce_mean(v)
            train_meter.update(metric_and_loss)

            # if (self.global_step + 1) % (self.args.save_step*self.args.iter_per_ep) == 0:
            #     self.save_checkpoint(str(self.global_step) + '.pth')
            train_pbar.set_description("Metering:" + str(train_meter))
            train_time = train_timer.elapse(True)

            if (self.global_step + 1) % self.args.iter_per_ep == 0:
                logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, self.epoch + 1, self.args.epochs, train_time, train_meter.avg))
                if not isinstance(train_meter.avg, dict):
                    raise ValueError(type(train_meter.avg))
                tr_meter_log = {**train_meter.avg }
                self.log_dict_to_wandb({f'{k}_ep': v for k, v in tr_meter_log.items()})
                tr_meter_log.update({'time': train_time})
                metric['Epoch%s' % (self.epoch + 1)].update( {'train': tr_meter_log})

                if is_main_process():
                    self.update_metric_file(metric)

            if (self.global_step + 1) % self.args.save_step == 0:
                logger.warning("Saving model...")
                self.save_checkpoint(str(self.global_step) + '.pth')
                if not self.args.deepspeed:
                    self.save_checkpoint(self.last_checkpoint_filename)
            
            # T.cuda.empty_cache()
            if (self.global_step + 1) % self.args.eval_step == 0:
                logger.warning("Evaluating...")
                if eval_loader:
                    eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                        use_tqdm=use_tqdm, train_eval_input=inputs) # sample a single round training sample to test whether over-fitting
                    logger.info('[Rank %s] Valid Step: %d, Time: %s\n %s' %
                                (self.rank, self.global_step, eval_time, eval_meter.avg))
                    # Update metric with eval metrics
                    val_meter_log = {**eval_meter.avg}
                    self.log_dict_to_wandb({f'{k}_step': v for k, v in val_meter_log.items()})
                    self.log_dict_to_wandb({f'{k}_iter': v for k, v in {**train_meter.avg}.items()})
                    self.log_dict_to_wandb({'lr_paramgp0_iter': self.optimizer.param_groups[0]["lr"]})
                    val_meter_log.update({'time': eval_time})
                    metric['Step%s' % (self.global_step + 1)].update({'eval': val_meter_log})

                    # Save metric file
                    if is_main_process():
                        self.update_metric_file(metric)
                    # T.cuda.empty_cache()
                train_meter = Meter()
                train_timer = Timer()
            self.global_step += 1
            train_pbar.update(1)
        if self.global_step % self.args.save_step != 0:
            logger.warning("Saving model...")
            self.save_checkpoint(self.last_checkpoint_filename)

        if self.global_step % self.args.eval_step != 0:
            logger.warning("Evaluating...")
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                    use_tqdm=use_tqdm)
                logger.info('[Rank %s] Valid Step: %d, Time: %s\n %s' %
                            (self.rank, self.global_step, eval_time, eval_meter.avg))
                val_meter_log = {**eval_meter.avg}
                self.log_dict_to_wandb({f'{k}_step': v for k, v in val_meter_log.items()})
                val_meter_log.update({'time': eval_time})
                metric['Step%s' % (self.global_step)].update({'eval': val_meter_log})

                # Save metric file
                if is_main_process():
                    self.update_metric_file(metric)
