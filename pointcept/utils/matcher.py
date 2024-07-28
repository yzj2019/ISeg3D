'''
matcher, modified from mask3d, detr, 3detr
https://github.com/JonasSchult/Mask3D/blob/main/models/matcher.py
https://github.com/facebookresearch/detr/blob/master/models/matcher.py
'''
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast



def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example
                (0 for the negative class and 1 for the positive class).
        targets: A float tensor with the same shape[1] as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)              # 两两的相交的面积
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]        # 广播, 两两的面积和
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule



def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example
                (0 for the negative class and 1 for the positive class).
        targets: A float tensor with the same shape[1] as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



# linear_sum_assignment using costmap
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
        cost_class: float = 1,
        cost_focal: float = 1,
        cost_dice: float = 1,
        instance_ignore_label: int = -1
    ):
        '''
        val set && random clicks, for matching pred instances and gt instances; unbatched
        '''
        super().__init__()
        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice
        self.instance_ignore_label = instance_ignore_label

        assert (
            cost_class != 0 or cost_focal != 0 or cost_dice != 0
        ), "all costs cant be 0"
            

    @torch.no_grad()
    def forward(self, pred, target):
        """Performs the matching

        Params:
            pred: This is a dict that contains at least these entries:
                 "cls_prob": Tensor of dim [N_pred_instance], the classification prob after normalization
                 "masks_heatmap": Tensor of dim [N_points, N_pred_instance], the predicted masks heatmap

            target: This is a dict containing:
                 "labels": Tensor of dim [N_target_instance] (where N_target_instance is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [N_points, N_target_instance] containing the target masks

        Returns:, 
        - idx_pred: int tensor, the indices of the selected predictions (in order)
        - idx_target: int tensor, is the indices of the corresponding selected targets (in order)
        - It holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        N_pred_ins = pred["cls_prob"].shape[0]

        # batch size == 1
        out_prob = pred["cls_prob"]
        tgt_ids = target["labels"].clone().long()
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        filter_ignore = tgt_ids == self.instance_ignore_label
        tgt_ids[filter_ignore] = 0
        cost_class = -out_prob[:, tgt_ids]
        cost_class[:, filter_ignore] = -1.0     # for ignore classes pretend perfect match ;) TODO better worst class match?

        out_mask = pred["masks_heatmap"].T      # [N_pred_instance, N_points]
        tgt_mask = target["masks"].T            # [N_target_instance, N_points]
        with autocast(enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            cost_focal = sigmoid_ce_loss_jit(out_mask, tgt_mask)
            # Compute the dice loss betwen masks
            cost_dice = dice_loss_jit(out_mask, tgt_mask)
        
        # Final cost matrix
        C = (
            self.cost_focal * cost_focal
            + self.cost_class * cost_class
            + self.cost_dice * cost_dice
        )
        C = C.reshape(N_pred_ins, -1).cpu().numpy()         # [N_pred_instance, N_target_instance]
        idx_pred, idx_target = linear_sum_assignment(C)     # best assignment for each gt instance

        return torch.as_tensor(idx_pred, dtype=torch.int64), torch.as_tensor(idx_target, dtype=torch.int64)


    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_focal: {}".format(self.cost_focal),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
