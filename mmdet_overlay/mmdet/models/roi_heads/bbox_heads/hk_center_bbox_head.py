# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import Parameter
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from .hk_base_bbox_head import HKBaseBBoxHead

def roi_gat_module(net, act, x, adj, roi_masks, roi_nums, roi_bs_ids, default_roi_num=512):
    '''
    gcn, roi as node, batch process

    Args:
        x (tensor): [B*R, D]
        adj (tensor): [B*Default_R, Default_R]
        roi_masks (tensor): [B*Default_R], mask to raw roi_feat
        roi_nums (list): [B], roi num per img
        roi_bs_ids: [B*R]
    Returns:
        out (tensor): [B*R, D]
        adj (tensor): [B*R, Default_R]
    '''
    roi_masks = roi_masks.reshape(-1, 1).repeat(1, default_roi_num).to(adj.device)  # [B*Default_R, Default_R]
    adj = torch.masked_select(adj, roi_masks).reshape(-1, default_roi_num)  # [B*R, Default_R]
    adj = adj / adj.max(dim=1)[0].unsqueeze(1)  # row normalize
    out_adj=adj

    outs = []
    bs_ids = torch.unique(roi_bs_ids, sorted=True)
    for i, bs_id in enumerate(bs_ids):
        inds = (roi_bs_ids == bs_id.item())
        _inp = x[inds, :]
        _adj = adj[inds, :roi_nums[i]]

        out,__adj = net(_inp, _adj)
        out_adj[inds, :roi_nums[i]] = __adj
        out = act(out)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    return outs, out_adj


def get_L1_dist(roi_feat, roi_bs_ids, default_roi_num=512):
    '''
    compute pair wise L1 distance

    Args:
        roi_feat (tensor): [B*R, D]
        roi_bs_ids: [B*R]
        default_roi_num: if input roi less than default, pad it
    Returns:
        out (tensor): [B*Default_R, Default_R]
        roi_mask (tensor): [B*Default_R], mask to raw roi_feat
    '''
    l1_dists = []
    roi_masks = []
    roi_nums = []
    bs_ids = torch.unique(roi_bs_ids, sorted=True)

    for bs_id in bs_ids:
        inds = (roi_bs_ids == bs_id.item())
        l1_dist = torch.cdist(roi_feat[inds, :], roi_feat[inds, :], p=1)
        roi_num = len(torch.nonzero(inds))
        roi_mask = torch.ones(roi_num)
        if not roi_num == default_roi_num:
            pad_num = default_roi_num - roi_num
            pad = nn.ZeroPad2d(padding=(0, pad_num, 0, pad_num))
            l1_dist = pad(l1_dist)

            _pad = torch.zeros(pad_num)
            roi_mask = torch.cat((roi_mask, _pad))

        roi_masks.append(roi_mask)
        l1_dists.append(l1_dist)
        roi_nums.append(roi_num)

    output = torch.cat(l1_dists, dim=0)
    roi_masks = torch.cat(roi_masks, dim=0).type(dtype=torch.bool)

    return output, roi_masks, roi_nums


def roi_gcn_module(net, act, x, adj, roi_masks, roi_nums, roi_bs_ids, default_roi_num=512):
    '''
    gcn, roi as node, batch process

    Args:
        x (tensor): [B*R, D]
        adj (tensor): [B*Default_R, Default_R]
        roi_masks (tensor): [B*Default_R], mask to raw roi_feat
        roi_nums (list): [B], roi num per img
        roi_bs_ids: [B*R]
    Returns:
        out (tensor): [B*R, D]
        adj (tensor): [B*R, Default_R]
    '''
    roi_masks = roi_masks.reshape(-1, 1).repeat(1, default_roi_num).to(adj.device)  # [B*Default_R, Default_R]
    adj = torch.masked_select(adj, roi_masks).reshape(-1, default_roi_num)  # [B*R, Default_R]
    adj = adj / adj.max(dim=1)[0].unsqueeze(1)  # row normalize

    outs = []
    bs_ids = torch.unique(roi_bs_ids, sorted=True)
    for i, bs_id in enumerate(bs_ids):
        inds = (roi_bs_ids == bs_id.item())
        _inp = x[inds, :]
        _adj = adj[inds, :roi_nums[i]]

        out = act(net(_inp, _adj))
        # out,__adj = net(_inp, _adj)
        # out = act(out)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    return outs, adj


def rearrange_pred_gt(gt_mat, labels, adj_pred, roi_bs_ids):
    '''
    re arrange the prediction and ground truth
    when roi num is less than default setting, zero pad the pred and gt

    Args:
        gt_mat (tensor): [C, C]
        labels (tensor): [B*R]
        adj_pred (tensor): [B*R, Default_R]
        roi_bs_ids: [B*R]
    Returns:
        preds (tensor): [B*R, Default_R]
        gts (tensor): [B*R, Default_R]
    '''

    pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
    gt_mat = pad(gt_mat)  # add bg cls

    bs_ids = torch.unique(roi_bs_ids, sorted=True)
    default_roi_num = adj_pred.size(1)
    preds = []
    gts = []
    for i, bs_id in enumerate(bs_ids):
        inds = (roi_bs_ids == bs_id.item())
        roi_num = len(torch.nonzero(inds))

        _label = labels[inds]
        ind1 = _label.reshape(-1, 1).repeat(1, roi_num)
        ind2 = _label.reshape(1, -1).repeat(roi_num, 1)
        _gt = gt_mat[ind1, ind2]  # [R,R]
        _pred = adj_pred[inds, :]  # [R,Default_R]

        if not roi_num == default_roi_num:
            pad_num = default_roi_num - roi_num
            pad = nn.ZeroPad2d(padding=(0, pad_num, 0, 0))
            _gt = pad(_gt)  # [R,Default_R]HKCenterGATHead
            _pred[:, -pad_num:] = 0

        preds.append(_pred)
        gts.append(_gt)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0).to(preds.device)

    return preds, gts


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    implement the function of: Y = A·(X·W)
        input X: [N,D1]
        weight W: [D1,D2]
         --> (X·W): [N,D2]
        adjacency matrix A: [N,N]
        output Y: [N, D2]
    N is the roi number, D1 is the roi feature dimension, D2 is the latent feature dimension
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # add weight and bias into the model as trainable Parameter
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adjacency):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adjacency, support)       #  x
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903

    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat


        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """


        """
        inp = F.dropout(inp, p=self.dropout, training=self.training)
        Wh = torch.mm(inp, self.W)  # [N, out_features]
        N = Wh.size()[0]
        #calcucate Eij edge
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # e =e+self.leakyrelu(adj)

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0.2, e, zero_vec)  # [N, N]



        # attention =attention+adj

        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)  # [N, N].[N, out_features] => [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


@HEADS.register_module()
class HKCenterBBoxHead(HKBaseBBoxHead):
    r"""Hybird knowledge head, utilize the co-occurrence information

    """

    def __init__(self,
                 num_shared_fcs=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_cls_convs=0,
                 num_cls_fcs=1,
                 num_reg_convs=0,
                 num_reg_fcs=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 roi_num=512,
                 cls_occur_path=None,
                 reg_spatial_path=None,
                 loss_adj=dict(type='MSELoss', loss_weight=1.0),
                 *args,
                 **kwargs):
        super(HKCenterBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs

        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.loss_adj = build_loss(loss_adj)
        self.default_roi_num = roi_num

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                0, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # load adj
        with open(cls_occur_path, 'rb') as f:
            cls_occur_info = pickle.load(f)
        self.cls_occur_mat = torch.from_numpy(cls_occur_info['A_img_lvl_eye']).float()
        with open(reg_spatial_path, 'rb') as f:
            reg_spatial_info = pickle.load(f)
        self.reg_spatial_mat = torch.from_numpy(reg_spatial_info['spatial_adj']).float()

        # add mlp for adj, 512 is the roi number
        self.cls_adj_fcs = nn.ModuleList()
        self.cls_adj_fcs.append(nn.Linear(self.default_roi_num, self.default_roi_num))
        self.cls_adj_fcs.append(nn.Linear(self.default_roi_num, self.default_roi_num))
        self.reg_adj_fcs = nn.ModuleList()
        self.reg_adj_fcs.append(nn.Linear(self.default_roi_num, self.default_roi_num))
        self.reg_adj_fcs.append(nn.Linear(self.default_roi_num, self.default_roi_num))

        # add gcn for specific branch
        self.gcn_relu = nn.LeakyReLU(0.2)
        self.cls_gcn = GraphAttentionLayer(self.fc_out_channels, self.fc_out_channels)
        self.reg_gcn = GraphAttentionLayer(self.fc_out_channels, self.fc_out_channels)

        # shortcut fc
        # self.cls_shortcut_fc = nn.Linear(self.fc_out_channels*2, self.fc_out_channels)
        # self.reg_shortcut_fc = nn.Linear(self.fc_out_channels*2, self.fc_out_channels)

        self.relu = nn.ReLU()

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.fc_out_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.fc_out_channels,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs'),
                        dict(name='cls_adj_fcs'),
                        dict(name='reg_adj_fcs'),
                        dict(name='cls_gcn'),
                        dict(name='reg_gcn'),
                        # dict(name='cls_shortcut_fc'),
                        # dict(name='reg_shortcut_fc')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, rois):
        roi_bs_ids = rois[:, 0]
        # shared part
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # roi pair wise L1 distance
        cls_adj, roi_masks, roi_nums = get_L1_dist(x_cls, roi_bs_ids, self.default_roi_num)  # [bs*Default_R,Default_R]
        reg_adj, roi_masks, roi_nums = get_L1_dist(x_reg, roi_bs_ids, self.default_roi_num)  # [bs*Default_R,Default_R]
        # mlp for adj
        for fc in self.cls_adj_fcs:
            cls_adj = self.relu(fc(cls_adj))
        for fc in self.reg_adj_fcs:
            reg_adj = self.relu(fc(reg_adj))
        # gcn
        cls_enh_feat, cls_adj_pred = roi_gcn_module(self.cls_gcn, self.gcn_relu, x_cls, cls_adj, roi_masks, roi_nums,
                                                    roi_bs_ids, self.default_roi_num)
        reg_enh_feat, reg_adj_pred = roi_gcn_module(self.reg_gcn, self.gcn_relu, x_reg, reg_adj, roi_masks, roi_nums,
                                                    roi_bs_ids, self.default_roi_num)
        # fusion
        x_cls = x_cls * cls_enh_feat
        x_reg = x_reg * reg_enh_feat

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, cls_adj_pred, reg_adj_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'cls_adj_pred', 'reg_adj_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             cls_adj_pred,
             reg_adj_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if cls_adj_pred is not None:
            cls_adj_pred, cls_adj_gt = rearrange_pred_gt(self.cls_occur_mat, labels, cls_adj_pred, rois[:, 0])
            losses['loss_cls_adj'] = self.loss_adj(cls_adj_pred, cls_adj_gt)
        if reg_adj_pred is not None:
            reg_adj_pred, reg_adj_gt = rearrange_pred_gt(self.reg_spatial_mat, labels, reg_adj_pred, rois[:, 0])
            losses['loss_reg_adj'] = self.loss_adj(reg_adj_pred, reg_adj_gt)
        return losses


