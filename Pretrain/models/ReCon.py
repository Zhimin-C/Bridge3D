import torch
import torch.nn as nn
import numpy as np
from .build import MODELS
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import Group, Encoder, TransformerEncoder, TransformerDecoder
from models.CrossModal import TextTransformer as TextEncoder
from models.CrossModal import VisionTransformer as ImageEncoder
from timm.models.layers import trunc_normal_
from utils.criterion import NCELoss, semantic_NCELoss

# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, points, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
        pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3:], bool_masked_pos


class ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity, select):
        B = similarity.size(0)
        losses = torch.zeros(B).to(similarity.device)
        for i in range(B):
            pos = torch.masked_select(similarity[i], select[i] == 1)
            neg = torch.masked_select(similarity[i], select[i] == 0)
            pos = torch.mean(pos, dim=0, keepdim=True)
            logits = torch.cat((pos, neg)).reshape(1, -1)
            label = torch.zeros(1, dtype=torch.long).to(similarity.device)
            losses[i] = self.criterion(logits/self.temperature, label)
        loss = losses.mean()
        return loss


@MODELS.register_module()
class ReCon(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[ReCon] ', logger='ReCon')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.drop_ratio = config.drop_ratio

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[ReCon] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='ReCon')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        # cross model contrastive
        self.csc_loss = torch.nn.SmoothL1Loss()
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False

        if self.csc_img:
            self.img_encoder = ImageEncoder(config)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
            self.img_proj = nn.Linear(self.trans_dim, 1024)
            self.img_proj.apply(self._init_weights)

        if self.csc_text:
            self.text_encoder = TextEncoder(config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_proj = nn.Linear(self.trans_dim, 512)
            self.text_proj.apply(self._init_weights)

        # single modal contrastive
        self.smc = config.self_contrastive
        if self.smc:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.trans_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128)
            )
            self.cls_proj.apply(self._init_weights)
            self.contrastive_head = ContrastiveHead(temperature=0.1)


        self.down_dim = 64

        self.rec_head_2d = nn.Conv1d(256, 1024, 1)
        self.rec_head_txt = nn.Conv1d(256, 512, 1)
        
        # self.rec_head_2d = nn.Sequential(nn.Linear(self.trans_dim, 1024),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(1024, 1024))
        # self.rec_head_txt = nn.Sequential(nn.Linear(self.trans_dim, 512),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(512, 512))
        
        
        
        self.up_sampling = nn.Upsample(scale_factor=14, mode="bilinear", align_corners=True)

        # self.down_head_2d = nn.Conv1d(1024, 256, 1)
        # self.down_head_txt = nn.Conv1d(512, self.down_dim, 1)
        self.nce_loss = NCELoss(temperature= 0.07)
        self.sem_nce_loss = semantic_NCELoss(temperature= 0.07)
        # self.down_head_2d = nn.Sequential(nn.Conv2d(1024, self.down_dim, 1),
        #                                   nn.Upsample(scale_factor=14, mode="bilinear", align_corners=True))

        # nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, dim),
        # )





    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts, data, **kwargs):
        B, pts_N = pts.shape[0], pts.shape[1]
        losses = {}
        neighborhood, center, fps_idx = self.group_divider(pts)
        cls_token, img_token, text_token, x_vis, mask = self.MAE_encoder(pts, neighborhood, center)

        paired_imgpose, img, pred_mask, pred_sem, caption_feat, img_scene_feat, img_obj_feat, psd_pred = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4].cuda(), data[5].cuda(), data[6].cuda(), data[7].cuda()
        # img_obj_feat = self.down_head_2d(img_obj_feat)
        img_obj_feat = self.up_sampling(img_obj_feat)

        psd_pred = psd_pred.view(-1)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)


        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec, x_rec_vis = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        # losses['mdm'] = self.loss_func(rebuild_points, gt_points)

        # obj level feat

        mask_index_max = pred_mask.max() + 1

        pred_mask = (
                torch.arange(
                    0,
                    pred_mask.shape[0] * int(mask_index_max),
                    mask_index_max,
                    device=pred_mask.device,
                )[:, None, None] + pred_mask
        )
        fps_idx = fps_idx.type(torch.int64)
        vis_visfps_idx = fps_idx[~mask].view(x_vis.shape[0], -1)
        mask_fps_idx = fps_idx[mask].view(x_vis.shape[0], -1)
        pairing_maskfps_idx = torch.zeros(B * mask_fps_idx.shape[1])
        pairing_visfps_idx = torch.zeros(B * vis_visfps_idx.shape[1])
        pairing_images = torch.zeros(B * pts_N, 3)

        for i in range(B):
            pairing_images[pts_N * i: pts_N * (i+1), 1:3] = paired_imgpose[i]
            pairing_images[pts_N * i: pts_N * (i + 1), 0] = i
            pairing_visfps_idx[vis_visfps_idx.shape[1] * i : vis_visfps_idx.shape[1] * (i + 1)] = vis_visfps_idx[i] + pts_N * i
            pairing_maskfps_idx[mask_fps_idx.shape[1] * i : mask_fps_idx.shape[1] * (i + 1)] = mask_fps_idx[i] + pts_N * i

        pred_mask_I = pred_mask.flatten()
        pred_sem = pred_sem.flatten()
        idx_P = torch.arange(pairing_visfps_idx.shape[0], device= pred_mask.device)
        total_pixels = pred_mask_I.shape[0]
        idx_I = torch.arange(total_pixels, device=pred_mask.device)

        pairing_visfps_idx = pairing_visfps_idx.type(torch.int64)
        pairing_images = torch.index_select(pairing_images, 0, pairing_visfps_idx)
        m = tuple(pairing_images.cpu().T.long())



        masked_pairing_sem_label = psd_pred[pairing_maskfps_idx.long()]
        mask_fg_num = (masked_pairing_sem_label != 0).sum()

        drop_num = int(self.drop_ratio * center.shape[1]) * center.shape[0]

        bg_indx = (masked_pairing_sem_label==0).nonzero()
        fg_indx = (masked_pairing_sem_label!=0).nonzero()
        bg_kpt_idx = torch.randperm(bg_indx.shape[0])[:(masked_pairing_sem_label.shape[0] - drop_num - mask_fg_num)]

        if mask_fg_num > (masked_pairing_sem_label.shape[0] - drop_num):
            kept_idx = fg_indx
        else:
            kept_idx = torch.cat((fg_indx, bg_indx[bg_kpt_idx]), 0)[:,0]

        # loss1 = self.loss_func(rebuild_points[kept_idx, :, :], gt_points[kept_idx, :, :])
        # losses['mdm'] = self.loss_func(rebuild_points, gt_points)
        losses['mdm'] = self.loss_func(rebuild_points[kept_idx, :, :], gt_points[kept_idx, :, :])


        scannete_txt_features = torch.load('/home/zhiminc/fastscratch/neurips/new/scannet_ViT16_clip_text.pth').cuda()
        # scannete_txt_features = torch.gather(scannete_txt_features, 0, pairing_sem_label.type(torch.int64))

        scannete_txt_features = torch.gather(scannete_txt_features, 0, psd_pred[pairing_visfps_idx.long()].unsqueeze(-1).repeat(1, 512))





#         with torch.no_grad():
#             one_hot_P = torch.sparse_coo_tensor(
#                 torch.stack((
#                     pred_mask[m], idx_P
#                 ), dim=0),
#                 torch.ones(idx_P.shape[0], device=pred_mask.device),
#                 (pred_mask.shape[0] * int(mask_index_max), idx_P.shape[0])
#             )

#             one_hot_I = torch.sparse_coo_tensor(
#                 torch.stack((
#                     pred_mask_I, idx_I
#                 ), dim=0),
#                 torch.ones(total_pixels, device=pred_mask.device),
#                 (pred_mask.shape[0] * int(mask_index_max), total_pixels)
#             )

#         pred_obj_2D_feats = self.rec_head_2d(x_rec_vis.reshape(-1, C).unsqueeze(-1)).view(-1, 1024)
        pred_obj_txt_feats = self.rec_head_txt(x_rec_vis.reshape(-1, C).unsqueeze(-1)).view(-1, 512)



#         k = one_hot_P @ pred_obj_2D_feats
#         k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
#         q = one_hot_I @ img_obj_feat.permute(0, 2, 3, 1).flatten(0, 2)
#         q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        # mask = torch.where(k[:, 0] != 0)
        # k = k[mask]
        # q = q[mask]

        # losses['obj_img'] = self.csc_loss(k, q.detach()).mean()

        # scannete_txt_features_dowm = self.down_head_txt(scannete_txt_features.unsqueeze(-1)).view(-1, self.down_dim)
        #
        # losses['obj_text'] = self.sem_nce_loss(pred_obj_txt_feats, scannete_txt_features_dowm, psd_pred[pairing_visfps_idx.long()])
        # if len(pred_obj_txt_feats[txt_mask]) == 0:
        #     loss_obj_txt = 0
        # else:
        txt_mask = psd_pred[pairing_visfps_idx.long()]!=0
        if len(pred_obj_txt_feats[txt_mask]) == 0:
            loss_obj_txt = 0
        else:
            losses['obj_text'] = self.csc_loss(pred_obj_txt_feats[txt_mask], scannete_txt_features[txt_mask].detach()).mean()
            
        # losses['obj_text'] = self.csc_loss(pred_obj_txt_feats, scannete_txt_features.detach()).mean()


        if self.csc_img:
            img_feature = img_scene_feat
            img_token = self.img_proj(img_token)
            losses['csc_img'] = self.csc_loss(img_feature, img_token).mean()

        if self.csc_text:
            text_feature = caption_feat
            text_token = self.text_proj(text_token)
            losses['csc_text'] = self.csc_loss(text_feature, text_token).mean()

        if self.smc:
            cls_proj = self.cls_proj(cls_token)
            cls_proj = nn.functional.normalize(cls_proj, dim=1)
            similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))

            select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
            for i in range(B):
                for j in range(B):
                    if text[i] == text[j]:
                        select[i, j] = 1
            losses['smc'] = self.contrastive_head(similarity, select)

        loss = sum(losses.values())
        return loss
