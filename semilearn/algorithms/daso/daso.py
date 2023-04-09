# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('daso')
class DASO(AlgorithmBase):

    """

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    # The set_hooks method in this code is responsible for registering hooks that will be called during the training process.
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            
            # obtain their respective logits and features by passing three kinds of data through the model
            # There are two different ways to process the data, depending on the value of 'self.use_cat'
            
            # self.use_cat is a boolean flag that determines how the input data should be processed in the train_step function.
            # If self.use_cat is True, the labeled samples (x_lb), weakly augmented unlabeled samples (x_ulb_w),
            # and strongly augmented unlabeled samples (x_ulb_s) are concatenated into a single tensor before being passed through the model.
            # If self.use_cat is False, the samples are processed separately, and gradients are not computed for the weakly augmented unlabeled samples.
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            # Finally, a feature dictionary feat_dict is created, containing the features for each type of input data:
            # (labeled samples, weakly augmented unlabeled samples, and strongly augmented unlabeled samples).
            
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean') # compute the supervised learning
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            # The original line using torch.softmax is commented out, as the custom compute_prob() method is used instead.
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute and creat mask (1 for sample with confidence above the threshold and 0 for samples below the threshold)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            # The use_hard_label flag determines whether the generated pseudo-labels should be hard integer labels or soft probability distributions.

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        # This line calls the process_out_dict method, which takes in the total loss (total_loss) and feature dictionary (feat_dict).
        # The method processes this information and returns a dictionary, out_dict, which may contain additional information or a specific structure required by the rest of the training process.
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        # This function takes in the supervised loss, unsupervised loss, total loss, and utility ratio (mask.float().mean().item()).
        # The method processes these values and returns a dictionary, log_dict, that contains the provided values in a specific structure,
        # which can be used for logging purposes (e.g., to visualize the training progress in TensorBoard).
        return out_dict, log_dict
        # Finally, the method returns both dictionaries: out_dict and log_dict.
        # These dictionaries can be used by the caller (e.g., the main training loop) to update the model's parameters and log the training progress.
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
        # In this case, get_argument returns a list of three SSL_Argument objects that represent the command-line arguments used for configuring the FixMatch algorithm. Each SSL_Argument takes three arguments:
        # A string representing the command-line flag (e.g., --hard_label, --T, and --p_cutoff).
        # A data type that the command-line argument should be converted to (e.g., str2bool, float).
        # A default value for the argument if it is not provided in the command line (e.g., True, 0.5, and 0.95).

    # These arguments allow users to configure the FixMatch algorithm's behavior when running the training script from the command line