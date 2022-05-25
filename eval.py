import os
import pdb
import sys
from tqdm import tqdm
from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from pytorch_msssim import ms_ssim, ssim
from scipy.spatial import ConvexHull
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
from typing_extensions import OrderedDict

import depth
import modules.generator as GEN
from animate import normalize_kp
from average_meter import AverageMeter
from fid_score import calculate_frechet_distance
from frames_dataset import TalkingHeadVideosDataset
from inception import InceptionV3
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback


def load_checkpoints(opt, config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    config['model_params']['common_params']['num_channels'] = 4
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path,map_location="cuda:0")

    ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, depth_encoder, depth_decoder, relative=True, adapt_movement_scale=True, cpu=False):
    sources = []
    drivings = []
    with torch.no_grad():
        predictions = []
        depth_gray = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            source = source.cuda()
            driving = driving.cuda()
        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]

        outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
        depth_driving = outputs[("disp", 0)]
        source_kp = torch.cat((source,depth_source),1)
        driving_kp = torch.cat((driving[:, :, 0],depth_driving),1)

        kp_source = kp_detector(source_kp)
        kp_driving_initial = kp_detector(driving_kp)

        # kp_source = kp_detector(source)
        # kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]

            if not cpu:
                driving_frame = driving_frame.cuda()
            outputs = depth_decoder(depth_encoder(driving_frame))
            depth_map = outputs[("disp", 0)]

            gray_driving = np.transpose(depth_map.data.cpu().numpy(), [0, 2, 3, 1])[0]
            gray_driving = 1-gray_driving/np.max(gray_driving)

            frame = torch.cat((driving_frame,depth_map),1)
            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm,source_depth = depth_source, driving_depth = depth_map)

            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(torch.cat((driving_frame, out['prediction']), dim=3).data.cpu().numpy(), [0, 2, 3, 1])[0])
            depth_gray.append(gray_driving)
    return sources, drivings, predictions,depth_gray


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num



def evaluate(eval_loader, generator, kp_detector, depth_encoder, depth_decoder, relative, adapt_movement_scale):
    l1_avg_meter = AverageMeter()
    mse_avg_meter = AverageMeter()
    ssim_avg_meter = AverageMeter()
    ms_ssim_avg_meter = AverageMeter()
    all_fakes = []
    all_reals = []
    n_samples = 0
    B = 8
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx])
    fid_model = fid_model.cuda()
    fid_model.eval()
    save_video = True
    with torch.no_grad():
        for i, val_batch in tqdm(enumerate(eval_loader), total=len(eval_loader), leave=False, desc='Forward'):
            video = val_batch[0]  # (video, video_name)
            video_name = val_batch[1][0]
            N, T, C, H, W = video.size()
            assert N == 1
            n_batch = (T - 1) // B + 1
            outputs = []
            for b_idx in tqdm(range(n_batch)):
                source = video[0, 0:1, :, :, :]
                driving = video[0, 1+b_idx*B:1+b_idx*B+B, :, :, :]

                if driving.size(0) < 1:
                    break
                source = source.cuda(non_blocking=True)
                driving = driving.cuda(non_blocking=True)
                source = source.repeat(driving.size(0), 1, 1, 1)

                source_depth = depth_decoder(depth_encoder(source)) # [("disp", 0)]
                source_rgbd = torch.cat([source, source_depth], dim=1)
                kp_source = kp_detector(source_rgbd)

                driving_depth = depth_decoder(depth_encoder(driving)) # [("disp", 0)]
                driving_rgbd = torch.cat([driving, driving_depth], dim=1)
                kp_driving = kp_detector(driving_rgbd)

                out = generator(source, kp_source=kp_source, kp_driving=kp_driving, source_depth=source_depth, driving_depth=driving_depth)

                prediction = out['prediction']

                if save_video:
                    out_imgs = torch.cat((source, driving, out['prediction']), dim=3)
                    out_imgs = np.transpose(out_imgs.data.cpu().numpy(), [0, 2, 3, 1])
                    out_imgs = [img_as_ubyte(img) for img in list(out_imgs)]
                    outputs += out_imgs

                all_fakes.append(prediction.cpu())
                all_reals.append(driving.cpu())
                n_samples += driving.size(0)

                l1_error = F.l1_loss(prediction, driving).item()
                mse_error = F.mse_loss(prediction, driving).item()
                ssim_score = ssim(prediction, driving, data_range=1.0, size_average=True).item()
                ms_ssim_score = ms_ssim(prediction, driving, data_range=1.0, size_average=True).item()
                l1_avg_meter.update(l1_error)
                mse_avg_meter.update(mse_error)
                ssim_avg_meter.update(ssim_score)
                ms_ssim_avg_meter.update(ms_ssim_score)

            if save_video:
                output_path = os.path.join(opt.results_dir, video_name)
                imageio.mimsave(output_path, outputs, fps=30)
 
        pred_acc_fake, pred_acc_real = np.empty((n_samples, 2048)), np.empty((n_samples, 2048))
        start = 0
        for fake, real in tqdm(zip(all_fakes, all_reals), total=len(all_fakes), desc='FID Inception'):
            pred_fake = fid_model(fake.cuda(non_blocking=True))[0]
            pred_real = fid_model(real.cuda(non_blocking=True))[0]
            if pred_fake.size(2) != 1 or pred_fake.size(3) != 1:
                pred_fake = F.adaptive_avg_pool2d(pred_fake, output_size=(1, 1))
            if pred_real.size(2) != 1 or pred_real.size(3) != 1:
                pred_real = F.adaptive_avg_pool2d(pred_real, output_size=(1, 1))
            end = start + pred_fake.size(0)
            pred_acc_fake[start:end] = pred_fake.cpu().data.numpy().reshape(pred_fake.size(0), -1)
            pred_acc_real[start:end] = pred_real.cpu().data.numpy().reshape(pred_real.size(0), -1)

        mu1 = np.mean(pred_acc_fake, axis=0)
        sigma1 = np.cov(pred_acc_fake, rowvar=False)
        mu2 = np.mean(pred_acc_real, axis=0)
        sigma2 = np.cov(pred_acc_real, rowvar=False)

        try:
            fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        except Exception:
            fid_score = 1000.0
        print(f"DaGAN results l1 {l1_avg_meter.avg} mse {mse_avg_meter.avg} ssim {ssim_avg_meter.avg} msssim {ms_ssim_avg_meter.avg} fid {fid_score}")

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-adv-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='pretrained/DaGAN_vox_adv_256.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--results_dir", default='results', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--generator", type=str, default='DepthAwareGenerator')
    parser.add_argument("--kp_num", type=int, default=15)


    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most aligned with source. (Only for faces, requires face_alignment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    os.makedirs(opt.results_dir, exist_ok=True)
    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('pretrained/depth_face_model/encoder.pth')
    loaded_dict_dec = torch.load('pretrained/depth_face_model/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    if not opt.cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()

    generator, kp_detector = load_checkpoints(opt, opt.config, opt.checkpoint)

    eval_dataset = TalkingHeadVideosDataset(is_train=False, root_dir='/D_data/Front/data/TalkingHead-1KH')
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, num_workers=8, drop_last=True, shuffle=False)


    evaluate(eval_loader, generator, kp_detector, depth_encoder, depth_decoder, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)

