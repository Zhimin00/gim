# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import torch
import numpy as np
import pytorch_lightning as pl
import os
import sys
import pdb
mast3r_path = os.path.abspath("/home/jovyan/workspace/mast3r-roma/mast3r")
dust3r_path = os.path.abspath("/home/jovyan/workspace/mast3r-roma/mast3r/dust3r")
sys.path.insert(0, mast3r_path)
sys.path.insert(0, dust3r_path)
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference, inference_warp, loss_of_one_batch, make_batch_symmetric
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3R_warp, AsymmetricMASt3R_only_warp
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
from dust3r_visloc.datasets.utils import get_HW_resolution
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from dust3r.utils.geometry import geotrf
from dust3r.utils.device import collate_with_cat
from pathlib import Path
from collections import OrderedDict

from tools.comm import all_gather
from tools.misc import lower_config, flattenList
from tools.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors

import torch.nn.functional as F

def dense_match(corresps, black_mask):
    im_A_to_im_B = corresps[1]["flow"]
    im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
    _, h, w, _ = im_A_to_im_B.shape
    b = 1
    low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(h, w), align_corners=False, mode="bilinear"
                )
    cert_clamp = 0
    factor = 0.5
    low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
    certainty = corresps[1]["certainty"] - low_res_certainty
    
    im_A_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=im_A_to_im_B.device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=im_A_to_im_B.device),
        ),
        indexing='ij'
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    im_A_coords = im_A_coords[None].expand(b, 2, h, w)
    certainty = certainty.sigmoid()  # logits -> probs
    im_A_coords = im_A_coords.permute(0, 2, 3, 1)
    if (im_A_to_im_B.abs() > 1).any() and True:
        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        certainty[wrong[:, None]] = 0
    certainty[black_mask] = 0
    im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    A_to_B, B_to_A = im_A_to_im_B.chunk(2)
    q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
    im_B_coords = im_A_coords
    s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
    warp = torch.cat((q_warp, s_warp), dim=2)
    certainty = torch.cat(certainty.chunk(2), dim=3)

    return (warp[0], certainty[0,0])

def kde(x, std = 0.1, down = None):
    # use a gaussian kernel to estimate density
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def sample_to_sparse(dense_matches,
            dense_certainty,
            num=10000,
            sample_mode = "threshold_balanced",
    ):
        if "threshold" in sample_mode:
            upper_thresh = 0.05
            dense_certainty = dense_certainty.clone()
            dense_certainty_ = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        matches, certainty = (
            dense_matches.reshape(-1, 4),
            dense_certainty.reshape(-1),
        )
        # noinspection PyUnboundLocalVariable
        certainty_ = dense_certainty_.reshape(-1)
        expansion_factor = 4 if "balanced" in sample_mode else 1
        if not certainty.sum(): certainty = certainty + 1e-8
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * num, len(certainty)),
                                         replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        good_certainty_ = certainty_[good_samples]
        good_certainty = good_certainty_
        if "balanced" not in sample_mode:
            return good_matches, good_certainty

        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                                             num_samples = min(num,len(good_certainty)),
                                             replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]


def crop(img, crop):
    out_cropped_img = img.clone()
    to_orig = torch.eye(3, device=img.device)
    out_cropped_img = img[crop_slice(crop)]
    to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, to_orig

@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)

def fine_matching(query_views, map_views, model, device, max_batch_size, fast_nn_params):
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    matches_im_map, matches_im_query, matches_confs = [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        conf_list_ppi = [cc11, cc21]

        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(pp2, pp1, subsample_or_initxy1=8,
                                                                       **fast_nn_params)
        matches_confs_ppi = torch.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        device = map_views['to_orig'][ppi].device
        matches_im_map_ppi = torch.from_numpy(matches_im_map_ppi.copy()).float().to(device)
        device = query_views['to_orig'][ppi].device
        matches_im_query_ppi = torch.from_numpy(matches_im_query_ppi.copy()).float().to(device)
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi], matches_im_map_ppi, norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi], matches_im_query_ppi, norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        matches_confs.append(matches_confs_ppi)

    matches_im_map = torch.cat(matches_im_map, dim=0)
    matches_im_query = torch.cat(matches_im_query, dim=0)
    matches_confs = torch.cat(matches_confs, dim=0)
    return matches_im_query, matches_im_map, matches_confs

class Trainer(pl.LightningModule):

    def __init__(self, pcfg, tcfg, dcfg, ncfg):
        super().__init__()

        self.save_hyperparameters()
        self.pcfg = pcfg
        self.tcfg = tcfg
        self.ncfg = ncfg
        ncfg = lower_config(ncfg)

        detector = model = None
        if pcfg.weight == 'gim_dkm':
            from networks.dkm.models.model_zoo.DKMv3 import DKMv3
            detector = None
            model = DKMv3(None, 540, 720, upsample_preds=True)
            model.h_resized = 660
            model.w_resized = 880
            model.upsample_preds = True
            model.upsample_res = (1152, 1536)
            model.use_soft_mutual_nearest_neighbours = False
        elif pcfg.weight == 'gim_roma':
            from networks.roma.roma import RoMa
            detector = None
            model = RoMa(img_size=[672])
        elif pcfg.weight == 'gim_loftr':
            from networks.loftr.loftr import LoFTR as MODEL
            detector = None
            model = MODEL(ncfg['loftr'])
        elif pcfg.weight == 'gim_lightglue':
            from networks.lightglue.superpoint import SuperPoint
            from networks.lightglue.models.matchers.lightglue import LightGlue
            detector = SuperPoint({
                'max_num_keypoints': 2048,
                'force_num_keypoints': True,
                'detection_threshold': 0.0,
                'nms_radius': 3,
                'trainable': False,
            })
            model = LightGlue({
                'filter_threshold': 0.1,
                'flash': False,
                'checkpointed': True,
            })
        elif pcfg.weight == 'root_sift':
            detector = None
            model = None
        elif pcfg.weight == 'mast3r':
            detector = None
            model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            model = AsymmetricMASt3R.from_pretrained(model_name)  
        elif pcfg.weight == 'mast3r_onlywarp':
            detector = None
            model_path = '/home/jovyan/workspace/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_onlywarp_0417/checkpoint-last.pth'
            model = AsymmetricMASt3R_only_warp.from_pretrained(model_path)
        elif pcfg.weight == 'mast3r_warpdpt':
            detector = None
            model_path = '/home/jovyan/workspace/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_warpdpt_0417/checkpoint-last.pth'
            model = AsymmetricMASt3R_warp.from_pretrained(model_path)


        self.detector = detector
        self.model = model

        checkpoints_path = ncfg['loftr']['weight']
        if ncfg['loftr']['weight'] is not None:
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']

            if pcfg.weight == 'gim_dkm':
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
                    if 'encoder.net.fc' in k:
                        state_dict.pop(k)
            elif pcfg.weight == 'gim_roma':
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            elif pcfg.weight == 'gim_lightglue':
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict.pop(k)
                    if k.startswith('superpoint.'):
                        state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
                self.detector.load_state_dict(state_dict)
                state_dict = torch.load(checkpoints_path, map_location='cpu')
                if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('superpoint.'):
                        state_dict.pop(k)
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

            self.model.load_state_dict(state_dict)
            print('Load weights {} success'.format(ncfg['loftr']['weight']))

    def compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.tcfg)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(batch['scene_id'], *batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'covisible0': batch['covisible0'],
            'covisible1': batch['covisible1'],
            'Rot': batch['Rot'],
            'Tns': batch['Tns'],
            'Rot1': batch['Rot1'],
            'Tns1': batch['Tns1'],
            't_errs2': batch['t_errs2'],
        }
        return metrics

    def inference(self, data):
        if self.pcfg.weight == 'gim_dkm' or self.pcfg.weight == 'gim_roma':
            self.gim_dkm_inference(data)
        elif self.pcfg.weight == 'gim_loftr':
            self.gim_loftr_inference(data)
        elif self.pcfg.weight == 'gim_lightglue':
            self.gim_lightglue_inference(data)
        elif self.pcfg.weight == 'root_sift':
            self.root_sift_inference(data)
        elif self.pcfg.weight == 'mast3r':
            self.mast3r_inference(data)
        elif 'warp' in self.pcfg.weight:
            self.mast3r_inference_warp(data)

    def mast3r_inference_warp(self, data):
        batch = [({"img": data['color0'], "idx": 0, "instance": 0},
                 {"img": data['color1'], "idx": 1, "instance": 1})]
        batch = collate_with_cat(batch[:1])
        view1, view2 = make_batch_symmetric(batch)
        _, _ , corresps = self.model(view1, view2)
        
        #pdb.set_trace()
        #remove black pixel
        black_mask1 = (data['color0'][0, 0] < 0.03125) & (data['color0'][0, 1] < 0.03125) & (data['color0'][0, 2] < 0.03125)
        black_mask2 = (data['color1'][0, 0] < 0.03125) & (data['color1'][0, 1] < 0.03125) & (data['color1'][0, 2] < 0.03125)
        black_mask1 = F.interpolate(black_mask1.float()[None, None], size=tuple(data['color0'].shape[-2:]), mode='nearest').bool()
        black_mask2 = F.interpolate(black_mask2.float()[None, None], size=tuple(data['color1'].shape[-2:]), mode='nearest').bool()
        black_mask = torch.cat((black_mask1, black_mask2), dim=0).bool()
        
        dense_matches, dense_certainty = dense_match(corresps, black_mask)   
        sparse_matches, mconf = sample_to_sparse(dense_matches, dense_certainty, 5000)

        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]
        height0, width0 = data['imsize0'][0]
        height1, width1 = data['imsize1'][0]
        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

        b_ids = torch.where(mconf[None])[0]
        mask = mconf > 0

        data.update({
            'hw0_i': hw0_i,
            'hw1_i': hw1_i,
            'mkpts0_f': kpts0[mask],
            'mkpts1_f': kpts1[mask],
            'm_bids': b_ids,
            'mconf': mconf[mask],
        })

    def mast3r_inference(self, data):
        batch = [({"img": data['color0'], "idx": 0, "instance": 0},
                 {"img": data['color1'], "idx": 1, "instance": 1})]
        view1, view2 = collate_with_cat(batch[:1])
        pred1, pred2 = self.model(view1, view2)
        # device = data['color0'].device 
        # images = [(
        #             {"img": data['color0'], "idx": 0, "instance": 0},
        #             {"img": data['color1'], "idx": 1, "instance": 1},
        #         )]

        
        # output = inference(images, self.model, 'cuda', batch_size=1)

        # _, pred1 = output['view1'], output['pred1']
        # _, pred2 = output['view2'], output['pred2']

        desc1, desc2 = (
            pred1['desc'].squeeze(0),
            pred2['desc'].squeeze(0),
        )#H, W, DIM
        conf_list = [pred1['desc_conf'].squeeze(0), pred2['desc_conf'].squeeze(0)]
        
        fast_nn_params = dict(device='cuda', dist="dot", block_size=2**13)

        coarse_matches_im0, coarse_matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=2,
            **fast_nn_params
        ) # N, 2
        coarse_mconf = torch.minimum(
            conf_list[1][coarse_matches_im1[:, 1], coarse_matches_im1[:, 0]],
            conf_list[0][coarse_matches_im0[:, 1], coarse_matches_im0[:, 0]]
        )
        b_ids = torch.where(coarse_mconf[None])[0]
        mask = coarse_mconf > 0

        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]
        height0, width0 = data['imsize0'][0]
        height1, width1 = data['imsize1'][0] 
        offset = 0.5 
        if data.get('color_fine0') is None:
            pts0 = torch.from_numpy(coarse_matches_im0.copy()).float().to(device)
            pts1 = torch.from_numpy(coarse_matches_im1.copy()).float().to(device)
            kpts0 = (
                torch.stack(
                    (
                        (width0 / hw0_i[1]) * (pts0[..., 0] + offset),
                        (height0 / hw0_i[0]) * (pts0[..., 1] + offset),
                    ),
                    dim=-1,
                )
                - offset
            )
            kpts1 = (
                torch.stack(
                    (
                        (width1 / hw1_i[1]) * (pts1[..., 0] + offset),
                        (height1 / hw1_i[0]) * (pts1[..., 1] + offset),
                    ),
                    dim=-1,
                )
                - offset
            )
            data.update({
                'hw0_i': hw0_i,
                'hw1_i': hw1_i,
                'mkpts0_f': kpts0[mask],
                'mkpts1_f': kpts1[mask],
                'm_bids': b_ids,
                'mconf': coarse_mconf[mask],
            })
        else: 
            pts0 = coarse_matches_im0.copy()
            pts1 = coarse_matches_im1.copy()
            HW0 = data['color_fine0'].shape[1:3] #B,H,W,3
            HW1 = data['color_fine1'].shape[1:3] 
            kpts0 = (
                np.stack(
                    (
                        (HW0[1] / hw0_i[1]) * (pts0[..., 0] + offset),
                        (HW0[0] / hw0_i[0]) * (pts0[..., 1] + offset),
                    ),
                    axis=-1,
                )
                - offset
            )
            kpts1 = (
                np.stack(
                    (
                        (HW1[1] / hw1_i[1]) * (pts1[..., 0] + offset),
                        (HW1[0] / hw1_i[0]) * (pts1[..., 1] + offset),
                    ),
                    axis=-1,
                )
                - offset
            )
            resized_img0, resized_img1 = data['color_fine0'][0], data['color_fine1'][0]
            crops1, crops2 = [], []
            to_orig1, to_orig2 = [], []
            query_resolution = get_HW_resolution(HW0[0], HW0[1], maxdim=512, patchsize=16)
            map_resolution = get_HW_resolution(HW1[0], HW1[1], maxdim=512, patchsize=16)
            for crop_q, crop_b, pair_tag in select_pairs_of_crops(resized_img1, resized_img0, kpts1,
                                                                            kpts0,
                                                                            maxdim=512,
                                                                            overlap=0.5,
                                                                            forced_resolution=[map_resolution,
                                                                                                query_resolution]):
                c1, trf1 = crop(resized_img1, crop_q)
                c2, trf2 = crop(resized_img0, crop_b)
                crops1.append(c1)
                crops2.append(c2)
                to_orig1.append(trf1)
                to_orig2.append(trf2)
            if len(crops1) == 0 or len(crops2) == 0:
                matches_im0, matches_im1, mconf = [], [], []
            else:
                crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
                if len(crops1.shape) == 3:
                    crops1, crops2 = crops1[None], crops2[None]
                to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
                map_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                        instance=['1' for _ in range(crops1.shape[0])],
                                        to_orig=to_orig1)
                query_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                        instance=['2' for _ in range(crops2.shape[0])],
                                        to_orig=to_orig2)

                # Inference and Matching
                matches_im0, matches_im1, mconf = fine_matching(query_crop_view,
                                                                map_crop_view,
                                                                self.model, device,
                                                                48,
                                                                fast_nn_params)
                query_to_orig_max = torch.tensor([[width0 / HW0[1], 0, 0],
                                [0, height0 / HW0[0], 0],
                                [0, 0, 1]], device = device)
                map_to_orig_max = torch.tensor([[width1 / HW1[1], 0, 0],
                                [0, height1 / HW1[0], 0],
                                [0, 0, 1]], device = device)
                matches_im0= geotrf(query_to_orig_max, matches_im0, norm=True)
                matches_im1 = geotrf(map_to_orig_max, matches_im1, norm=True)
                mask = mconf > 0
                
                b_ids = torch.where(mconf[None])[0]
            data.update({
                'hw0_i': HW0,
                'hw1_i': HW1,
                'mkpts0_f': matches_im0,
                'mkpts1_f': matches_im1,
                'm_bids': b_ids,
                'mconf': mconf,
            })

        


    def gim_dkm_inference(self, data):
        dense_matches, dense_certainty = self.model.match(data['color0'], data['color1'])
        print(dense_matches.shape, dense_certainty.shape)
        pdb.set_trace()
        sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)
        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]
        height0, width0 = data['imsize0'][0]
        height1, width1 = data['imsize1'][0]
        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

        b_ids = torch.where(mconf[None])[0]
        mask = mconf > 0
        
        pdb.set_trace()
        print('kpts0.shape: ', kpts0.shape)
        print('kpts1.shape: ', kpts1.shape)
        print('b_ids.shape: ', b_ids.shape)
        print('mconf.shape: ', mconf.shape)
        print('mask.shape: ', mask.shape)
        data.update({
            'hw0_i': hw0_i,
            'hw1_i': hw1_i,
            'mkpts0_f': kpts0[mask],
            'mkpts1_f': kpts1[mask],
            'm_bids': b_ids,
            'mconf': mconf[mask],
        })

    def gim_loftr_inference(self, data):
        self.model(data)

    def gim_lightglue_inference(self, data):
        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]

        pred = {}
        pred.update({k+'0': v for k, v in self.detector({
            "image": data["image0"],
            "image_size": data["resize0"][:, [1, 0]],
        }).items()})
        pred.update({k+'1': v for k, v in self.detector({
            "image": data["image1"],
            "image_size": data["resize1"][:, [1, 0]],
        }).items()})
        pred.update(self.model({**pred, **data}))

        bs = data['image0'].size(0)
        mkpts0_f = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        mkpts1_f = torch.cat([kp * s for kp, s in zip(pred['keypoints1'], data['scale1'][:, None])])
        m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = pred['matches']
        mkpts0_f = torch.cat([mkpts0_f[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mkpts1_f = torch.cat([mkpts1_f[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
        m_bids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mconf = torch.cat(pred['scores'])

        data.update({
            'hw0_i': hw0_i,
            'hw1_i': hw1_i,
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
            'm_bids': m_bids,
            'mconf': mconf,
        })

    def root_sift_inference(self, data):
        # matching two images by sift
        image0 = data['color0'].squeeze().permute(1, 2, 0).cpu().numpy() * 255
        image1 = data['color1'].squeeze().permute(1, 2, 0).cpu().numpy() * 255

        image0 = cv2.cvtColor(image0.astype(np.uint8), cv2.COLOR_RGB2BGR)
        image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2BGR)

        H0, W0 = image0.shape[:2]
        H1, W1 = image1.shape[:2]

        sift0 = cv2.SIFT_create(nfeatures=H0*W0//64, contrastThreshold=1e-5)
        sift1 = cv2.SIFT_create(nfeatures=H1*W1//64, contrastThreshold=1e-5)

        kpts0, desc0 = sift0.detectAndCompute(image0, None)
        kpts1, desc1 = sift1.detectAndCompute(image1, None)
        kpts0 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts0])
        kpts1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts1])

        kpts0, desc0, kpts1, desc1 = map(lambda x: torch.from_numpy(x).cuda().float(), [kpts0, desc0, kpts1, desc1])
        desc0, desc1 = map(lambda x: (x / x.sum(dim=1, keepdim=True)).sqrt(), [desc0, desc1])

        matches = desc0 @ desc1.transpose(0, 1)

        mask = (matches == matches.max(dim=1, keepdim=True).values) & \
               (matches == matches.max(dim=0, keepdim=True).values)
        valid, indices = mask.max(dim=1)
        ratio = torch.topk(matches, k=2, dim=1).values
        # noinspection PyUnresolvedReferences
        ratio = (-2 * ratio + 2).sqrt()
        ratio = (ratio[:, 0] / ratio[:, 1]) < 0.8
        valid = valid & ratio

        kpts0 = kpts0[valid] * data['scale0']
        kpts1 = kpts1[indices[valid]] * data['scale1']
        mconf = matches.max(dim=1).values[valid]

        b_ids = torch.where(valid[None])[0]

        data.update({
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'mkpts0_f': kpts0,
            'mkpts1_f': kpts1,
            'm_bids': b_ids,
            'mconf': mconf,
        })

    def test_step(self, batch, batch_idx):
        self.inference(batch)
        metrics = self.compute_metrics(batch)
        return {'Metrics': metrics}

    def test_epoch_end(self, outputs):

        metrics = [o['Metrics'] for o in outputs]
        metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in metrics]))) for k in metrics[0]}

        unq_ids = list(OrderedDict((iden, i) for i, iden in enumerate(metrics['identifiers'])).values())
        ord_ids = sorted(unq_ids, key=lambda x:metrics['identifiers'][x])
        metrics = {k:[v[x] for x in ord_ids] for k,v in metrics.items()}
        # ['identifiers', 'epi_errs', 'R_errs', 't_errs', 'inliers',
        #  'covisible0', 'covisible1', 'Rot', 'Tns', 'Rot1', 'Tns1']
        output = ''
        output += 'identifiers covisible0 covisible1 R_errs t_errs t_errs2 '
        output += 'Bef.Prec Bef.Num Aft.Prec Aft.Num\n'
        eet = 5e-4  # epi_err_thr
        mean = lambda x: sum(x) / max(len(x), 1)
        for ids, epi, Rer, Ter, Ter2, inl, co0, co1 in zip(
                metrics['identifiers'], metrics['epi_errs'],
                metrics['R_errs'], metrics['t_errs'], metrics['t_errs2'], metrics['inliers'],
                metrics['covisible0'], metrics['covisible1']):
            bef = epi < eet
            aft = epi[inl] < eet
            output += f'{ids} {co0} {co1} {Rer} {Ter} {Ter2} '
            output += f'{mean(bef)} {sum(bef)} {mean(aft)} {sum(aft)}\n'

        scene = Path(self.hparams['dcfg'][self.pcfg["tests"]]['DATASET']['TESTS']['LIST_PATH']).stem.split('_')[0]
        path = f"dump/zeb/[T] {self.pcfg.weight} {scene:>15} {self.pcfg.version}.txt"
        with open(path, 'w') as file:
            file.write(output)
