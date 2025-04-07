'''
agile3d 的 mask decoder, 文章应该是参考的 Mask2Former, Mask3D 的设计

transformer layer, transformer decoder, mask decoder
'''

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import spconv.pytorch as spconv

import warnings
from typing import Tuple, Type, List, Dict
from collections.abc import Sequence

from .misc import FFNLayer, GenericMLP
from .attention import SqueezedCrossAttentionLayer, SqueezedSelfAttentionLayer, BatchedSelfAttentionLayer
from ..builder import MODELS, build_model
from ..utils import offset2batch

from pointcept.utils_iseg.clicker import MyClick

from .positional_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D


# ------------------------------------------------------------------------
# 参考 transformer.py, agile3d 的 decoder layer
# ------------------------------------------------------------------------

@MODELS.register_module()
class Mask3dDecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 1024,
        activation: str = "relu",
        skip_first_layer_pe: bool = False,
        attn_drop: float = 0., layer_drop: float = 0.,
        norm_first=False
    ) -> None:
        """
        A transformer block with three layers: 
        1. masked click to scene attention, using instance pred before to refine as mask
        2. click to click attention
        3. ffn layer

        Arguments:
        - embedding_dim (int): the channel dimension of the embeddings, same as click embedding_dim
        - num_heads (int): the number of heads in the attention layers
        - mlp_dim (int): the hidden dimension of the mlp block
        - activation (str): the activation name of the mlp block
        - skip_first_layer_pe (bool): skip the PE on the first layer
        - attn_drop: dropout percent in each attn layer, default 0.1
        - layer_drop: dropout percent in each dropout layer, default 0.1
        - norm_first: whether to normalize before attn (unused)
        """
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        self.masked_click_to_scene = SqueezedCrossAttentionLayer(
            embedding_dim, num_heads, attn_drop=attn_drop, layer_drop=layer_drop
        )

        self.click_to_click = BatchedSelfAttentionLayer(
            embedding_dim, num_heads, attn_drop=attn_drop, layer_drop=layer_drop
        )

        self.mlp = FFNLayer(embedding_dim, mlp_dim=mlp_dim, activation=activation, dropout=layer_drop, normalize_before=norm_first)


    def forward(
        self, scene: Tensor, clicks: Tensor, scene_pe: Tensor, clicks_pe: Tensor,  
        scene_batch, clicks_batch, attn_mask=None
    ) -> Tuple[Tensor, Tensor]:
        '''
        Args:
        - scene (torch.Tensor): scene pcd embedding. Shape as (N_points, embedding_dim) for any N_points.
        - clicks (torch.Tensor): click embedding. Shape as (N_clicks, 3, embedding_dim) for any N_clicks.
        - scene_pe (torch.Tensor): the positional encoding to add to the scene. Must have the same shape as scene.
        - clicks_pe (torch.Tensor): the positional encoding to add to the click. Must have the same shape as click.
        - scene_batch (torch.Tensor): shape as (N_points), which batch the point belongs to
        - clicks_batch (torch.Tensor): shape as (N_clicks), which batch the click belongs to
        - attn_mask (torch.Tensor, default None): mask for masked_click_to_scene layer. Shape as (N_clicks, N_points).

        Returns:
        - torch.Tensor: the processed scene embedding
        - torch.Tensor: the processed click embedding
        '''

        clicks = self.masked_click_to_scene(source=scene, query=clicks,
                                            source_batch=scene_batch, query_batch=clicks_batch, 
                                            attn_mask=attn_mask, 
                                            source_pe=None if self.skip_first_layer_pe else scene_pe, 
                                            query_pe=None if self.skip_first_layer_pe else clicks_pe)

        clicks = self.click_to_click(x=clicks, pe=None if self.skip_first_layer_pe else clicks_pe)

        # TODO 考虑加个 click.T -> click.T attention

        clicks = self.mlp(clicks)

        return scene, clicks



@MODELS.register_module()
class Mask3dMaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_block_cfg,
        feature_maps_dims: Sequence,
        clicks_embedding_dim: int,
        mask_head_hidden_dims: Sequence,
        cls_head_hidden_dims: Sequence,
        num_classes: int,
        with_attn_mask: Sequence,
        enable_final_block = False,
        return_block_id = 5,
        mask_num = 1,
    ) -> None:
        '''
        做数据shape的处理, 根据 scene embedding 和 click point 进行 click embedding的构建, 用transformer、head计算mask

        Args:
        - feature_maps_dims: 输入的 feature_maps 各自的 embedding_dim, sorted from coarse to fine, len(input_dims) == feature_maps_num, 有几张feature map就用几层 transformer block; in_proj_layers 用到
        - clicks_embedding_dim: int, in_proj_clicks 用
        - num_classes: 语义labels的类别数
        - with_attn_mask: 每层 transformer block 是否需要 attn_mask, same len as feature_maps_dims
        - enable_final_block: 是否对最顶层 feature map 也用 transformer block 去 query
        - return_block_id: 忽略id个blocks之后的, 用于 eval 输出不同层结果
        - mask_num: TODO 增加多尺度mask

        Return: mask_logits, cls_logits
        - mask_logits: 场景中点属于哪个click对应的mask, N_points x N_clicks
        - cls_logits: click点对应的类别, N_clicks x num_classes
        '''
        super().__init__()
        self.feature_maps_dims = feature_maps_dims
        self.feature_maps_num = len(feature_maps_dims)
        if self.training:
            self.return_block_id = self.feature_maps_num
        else:
            self.return_block_id = return_block_id
        self.embedding_dim = transformer_block_cfg.embedding_dim
        self.with_attn_mask = with_attn_mask
        self.enable_final_block = enable_final_block
        assert self.return_block_id >= 0, "can not return negative block result"
        assert len(with_attn_mask) == self.feature_maps_num, f"len(with_attn_mask) must be same as feature_maps_dims {self.feature_maps_dims}, but got{len(with_attn_mask)}"
        self.num_classes = num_classes

        # in_proj && transformer block
        self.in_proj_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.feature_maps_num-1):
            in_proj = nn.Linear(feature_maps_dims[i], self.embedding_dim)
            self.in_proj_layers.append(in_proj)
            transformer_block_cfg.skip_first_layer_pe = (i == 0)
            self.blocks.append(
                MODELS.build(transformer_block_cfg)
            )
        in_proj = nn.Linear(feature_maps_dims[-1], self.embedding_dim)
        self.in_proj_layers.append(in_proj)
        if enable_final_block:
            transformer_block_cfg.skip_first_layer_pe = (self.feature_maps_num==1)
            self.blocks.append(
                MODELS.build(transformer_block_cfg)
            )
        self.in_proj_clicks = nn.Linear(clicks_embedding_dim, self.embedding_dim)
        # positional encoding && pooling
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=self.embedding_dim,
            gauss_scale=1.0,
            normalize=True,
        )
        self.pooling = spconv.SparseAvgPool3d(kernel_size=2, stride=2)

        # head
        self.norm_before_head = nn.LayerNorm(self.embedding_dim)
        self.mask_head = GenericMLP(
            input_dim = self.embedding_dim,
            hidden_dims = mask_head_hidden_dims,
            output_dim = self.embedding_dim,
            norm_fn_name = None,
            activation = "relu",
            use_conv=False,
            dropout=0.,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform"
        )
        self.cls_head = GenericMLP(
            input_dim = self.embedding_dim,
            hidden_dims = cls_head_hidden_dims,
            output_dim = num_classes,
            norm_fn_name = None,
            activation = "relu",
            use_conv=False,
            dropout=0.,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform"
        )
        self.fus_head = GenericMLP(
            input_dim = 3,
            hidden_dims = [2],
            output_dim = 1,
            norm_fn_name = None,
            activation = "relu",
            use_conv=False,
            dropout=0.,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform"
        )


    def downsample(self, feat, scene_dict, num_pooling=0):
        '''downsample and sort feat with self.pooling, 参照 spunet 的 down up 的过程
            - feat: (N_points, c)
        '''
        assert num_pooling >= 0
        grid_coord = scene_dict["grid_coord"]
        offset = scene_dict["offset"]
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat.clone().float(),
            indices=torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1
        )
        for _ in range(num_pooling):
            x = self.pooling(x)
        feat, _ = self.sort_with_indices(x)
        return feat

    def sort_with_indices(self, x:spconv.SparseConvTensor):
        '''sort sparse tensor with it's indices, return it's features and indices as (N_points, c)'''
        y = torch.sparse_coo_tensor(
            indices=x.indices.T,            # indice always like (batch, x, y, z)
            values=x.features,
            size=(x.batch_size, *x.spatial_shape, x.features.shape[1]),
            device=x.features.device,
            requires_grad=x.features.requires_grad
        )
        y = y.coalesce()
        return y.values(), y.indices().T

    def transformer_fwd(self, clicks_embedding, clicks_pe, clicks_batch, scene_dict, scene_pe, last_masks_heatmap=None):
        '''
        transformer forward
        - clicks_embedding: query, N_clicks x 3 x embedding_dim
        - clicks_pe: N_clicks x 3 x embedding_dim
        - clicks_batch: (N_clicks,)
        - scene_dict: N_points data dict
        - feature_maps: sparse tensor, list len == self.feature_maps_num, sorted from coarse to fine, (j, N_points_j, self.feature_maps_dims[j])
        - scene_pe: N_points x embedding_dim
        - last_masks_heatmap: N_points x N_clicks, corresponding to clicks_embedding and clicks_pe
        '''
        feature_maps = scene_dict["feature_maps"]
        masks_heatmap = last_masks_heatmap.clone().detach()
        attn_mask = None
        # for i in range(self.depth):
            # depth 次共享权重的处理
            # TODO make next click, return 3 maskheatmaps, 把depth改到外头
        for j in range(self.feature_maps_num-1):
            # feature_maps_num-1 个 block
            feature_map = feature_maps[j]           # (N_points_j, self.feature_maps_dims[j])
            feature_map, indices = self.sort_with_indices(feature_map)
            scene_batch = indices[:, 0]
            feature_map = self.in_proj_layers[j](feature_map.float())   # (N_points_j, embedding_dim)
            attn_mask = masks_heatmap
            if self.with_attn_mask[j]:
                # 需要保证 last_masks_heatmap 不会限制下次推理, 所以 with_attn_mask[0] == False
                attn_mask = self.mask_module(masks_heatmap, N_clicks=clicks_embedding.shape[0])         # (N_points, N_clicks)
            attn_mask = self.downsample(attn_mask, scene_dict, num_pooling=self.feature_maps_num-1-j).T     # (N_clicks, N_points_j)
            attn_mask = torch.stack([attn_mask, attn_mask.clone().detach(), attn_mask.clone().detach()], dim=1) # (N_clicks, 3, N_points_j)
            # assert attn_mask.shape[0] == clicks_embedding.shape[0] and attn_mask.shape[1] == feature_map.shape[0], f"{attn_mask.shape}, {clicks_embedding.shape[0]}, {feature_map.shape[0]}, {j}"
                
            # after sort, scene and scene_pe become corresponding in point-wise
            scene_pe_down = self.downsample(scene_pe, scene_dict, num_pooling=self.feature_maps_num-1-j)    #(N_points, ...) -> (N_points_j, ...)
            scene_embedding, clicks_embedding = self.blocks[j](
                scene=feature_map, clicks=clicks_embedding,
                scene_pe=scene_pe_down, clicks_pe=clicks_pe,
                scene_batch=scene_batch, clicks_batch=clicks_batch,
                attn_mask=attn_mask
            )
            masks_heatmap, cls_logits = self.head(self.feature_used_for_masks, clicks_embedding)
                
            if self.return_block_id == j:
                break

        if self.enable_final_block:
            feature_map = feature_maps[-1].features           # (N_points, self.feature_maps_dims[j])
            feature_map = self.in_proj_layers[-1](feature_map)   # (N_points, embedding_dim)
            scene_batch = feature_maps[-1].indices[:, 0]        # 顺序与scene_dict相同, 不需要 sort_with_indices
            attn_mask = None
            if self.with_attn_mask[-1]:
                attn_mask = self.mask_module(masks_heatmap, N_clicks=clicks_embedding.shape[0]).T         # (N_clicks, N_points)
                attn_mask = torch.stack([attn_mask, attn_mask.clone().detach(), attn_mask.clone().detach()], dim=1) # (N_clicks, 3, N_points)
            scene_embedding, clicks_embedding = self.blocks[-1](
                scene=feature_map, clicks=clicks_embedding,
                scene_pe=scene_pe, clicks_pe=clicks_pe,
                scene_batch=scene_batch, clicks_batch=clicks_batch,
                attn_mask=attn_mask
            )
            masks_heatmap, cls_logits = self.head(scene_embedding, clicks_embedding)
                
        
        return masks_heatmap, cls_logits


    def get_scene_pe(self, scene_dict):
        '''
        获取scene的positional embedding
        '''
        offset = scene_dict["offset"]
        batch = offset2batch(offset)
        scene_pe = []
        for b in range(len(offset)):
            batch_mask = batch == b
            mins  = scene_dict["grid_coord"][batch_mask].min(dim=0)[0]
            maxs = scene_dict["grid_coord"][batch_mask].max(dim=0)[0]
            pe = self.pos_enc(
                scene_dict["grid_coord"][batch_mask].float(),
                input_range=[mins, maxs]
            )
            scene_pe.append(pe)
        return torch.cat(scene_pe)
    
    def get_clicks_index(self, clicks, scene_dict):
        '''
        获取click在scene中的index, 用 click.index在grid_coord中寻址
        - clicks[batch_id][class_tag]['init'] -> MyClick
        - scene_dict: 一般为fragment处理后的input_dict
        '''
        batch = offset2batch(scene_dict["offset"])
        grid_coord = torch.cat([batch.unsqueeze(-1).int(), scene_dict["grid_coord"].int()], dim=1).contiguous()
        for batch_id in range(len(clicks)):
            for class_tag in clicks[batch_id].keys():
                for name in clicks[batch_id][class_tag].keys():
                    click = clicks[batch_id][class_tag][name]
                    if click.index != None:
                        continue
                    mask_x = grid_coord[:,1] == click.coords[0]
                    mask_y = grid_coord[:,2] == click.coords[1]
                    mask_z = grid_coord[:,3] == click.coords[2]
                    mask_b = grid_coord[:,0] == click.batch_id
                    mask = torch.logical_and(torch.logical_and(mask_x, mask_y), torch.logical_and(mask_z, mask_b))
                    click.index = torch.where(mask)[0].item()
    
    def get_clicks_idxs(self, clicks, N_points):
        '''
        获取click的index list
        TODO 区分positive 和 negative, 不同click用不同batch? 一个click分3channel, initial click, pos click, neg click
        - clicks: 
            - clicks[batch_id][class_tag]['init'] -> MyClick, 见 pointcept.utils.clicker
            - 可用 click.index在grid_coord中寻址
        - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
        '''
        clicks_idx_list = []
        for batch_id in range(len(clicks)):
            for class_tag in clicks[batch_id].keys():
                init = clicks[batch_id][class_tag]['init']
                pos = clicks[batch_id][class_tag]['pos']
                neg = clicks[batch_id][class_tag]['neg']
                init, pos = init.index, pos.index
                if neg == None:
                    neg = N_points
                else:
                    neg = neg.index
                clicks_idx_list.extend([init, pos, neg])
        clicks_idxs = torch.Tensor(clicks_idx_list).long()
        return clicks_idxs

    def get_clicks_feat(self, clicks_idxs, scene_embedding):
        '''
        获取click的feature
        - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
        - scene_embedding (torch.Tensor): scene pcd embedding. Shape as N_points x embedding_dim for any N_points.
        - clicks_embedding: (torch.Tensor), N_clicks x 3 x embedding_dim
        '''
        scene_embedding = torch.cat([scene_embedding, torch.zeros((1, scene_embedding.shape[1])).to(scene_embedding.device)])
        return scene_embedding[clicks_idxs].detach().clone().reshape(-1, 3, scene_embedding.shape[1])
    
    def get_clicks_pe(self, clicks_idxs, scene_pe):
        '''
        获取click的positional embedding
        - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
        - scene_pe (torch.Tensor): scene positional embedding. Shape as N_points x embedding_dim for any N_points.
        - clicks_pe: (torch.Tensor), N_clicks x 3 x embedding_dim
        '''
        scene_pe = torch.cat([scene_pe, torch.zeros((1, scene_pe.shape[1])).to(scene_pe.device)])
        return scene_pe[clicks_idxs].detach().clone().reshape(-1, 3, scene_pe.shape[1])
    
    def get_clicks_batch(self, clicks):
        '''获取click属于哪个batch'''
        clicks_batch = []
        for batch_id in range(len(clicks)):
            clicks_batch.extend([batch_id for _ in clicks[batch_id].keys()])
        return torch.Tensor(clicks_batch).int()

    def mask_module(self, masks_heatmap, N_clicks, tau=0.5):
        '''构建 attn_mask, N_points x N_clicks -> N_points x N_clicks
            - 默认前 M_clicks 个就是上次的 clicks queries
        '''
        assert N_clicks == masks_heatmap.shape[1], "clicks number must be corresponding to the last masks number"
        attn_mask = torch.zeros((masks_heatmap.shape[0], N_clicks)).to(masks_heatmap.device)
        attn_mask[:, :masks_heatmap.shape[1]] = (masks_heatmap.detach() < tau)
        return attn_mask.detach()
    
    def head(self, scene_embedding, clicks_embedding):
        '''head
        - scene_embedding: N_points x embedding_dim
        - clicks_embedding: N_clicks x 3 x embedding_dim'''
        # get new mask
        # clicks_embedding = clicks_embedding[:, 0, :]        # 只用 init 生成mask
        clicks_embedding = self.norm_before_head(clicks_embedding)
        masks_embedding = self.mask_head(clicks_embedding)  # N_clicks x 3 x embedding_dim
        # masks_logits = scene_embedding @ masks_embedding.T  # embedding 相乘得到 N_click 个 mask, N_points x N_clicks
        masks_logits = (masks_embedding @ scene_embedding.T).permute([2,0,1])
        masks_logits = self.fus_head(masks_logits).squeeze(2)      # (N_points x N_clicks x 3) -> (N_points x N_clicks)
        masks_heatmap = masks_logits.sigmoid()              # element_wise 归一化
        # get new cls
        cls_logits = self.cls_head(clicks_embedding[:, 0, :])        # 可以考虑head之间压缩并行? N_clicks x num_classes
        return masks_heatmap, cls_logits


    def forward(self, scene_dict, clicks, last_masks_heatmap):
        '''
        - scene_dict: ['coord', 'grid_coord', 'index', 'offset', 'feat']
        - scene_embedding: the features of scene
        - clicks: list of clicks, N_clicks
        - last_masks_heatmap: N_points x N_clicks, each element in [0,1]
        '''
        # 数据准备, 获取embedding
        scene_dict["feature_maps"] = scene_dict["feature_maps"][-self.feature_maps_num:]
        feature_maps = scene_dict["feature_maps"]       # list of spconv.sparsetensor
        assert not torch.any(torch.isnan(feature_maps[-1].features)), f"feature_maps[-1].features is nan before in_proj!"
        self.feature_used_for_masks = self.in_proj_layers[-1](feature_maps[-1].features)        # 需要梯度反传
        assert not torch.any(torch.isnan(self.feature_used_for_masks)), f"self.feature_used_for_masks is nan after in_proj!"
        grid_coord = scene_dict["grid_coord"]
        # self.get_clicks_index(clicks, scene_dict)
        clicks_idxs = self.get_clicks_idxs(clicks, self.feature_used_for_masks.shape[0])
        clicks_embedding = self.get_clicks_feat(clicks_idxs, feature_maps[-1].features)     # 从最顶层feature map生成clicks feature
        clicks_embedding = self.in_proj_clicks(clicks_embedding)
        scene_pe = self.get_scene_pe(scene_dict)
        clicks_pe = self.get_clicks_pe(clicks_idxs, scene_pe)
        clicks_batch = self.get_clicks_batch(clicks).to(clicks_embedding.device)

        # 首次, 利用click embedding生成pred_logits
        if last_masks_heatmap == None:
            last_masks_heatmap, _ = self.head(self.feature_used_for_masks, clicks_embedding)
        # with torch.cuda.amp.autocast(enabled=True):          # 用了flash attention, 必须auto cast 到低精度
        masks_heatmap, cls_logits = self.transformer_fwd(
            clicks_embedding, clicks_pe, clicks_batch,
            scene_dict, scene_pe,
            last_masks_heatmap
        )
        
        return masks_heatmap, cls_logits