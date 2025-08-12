import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-label classification.
    
    This loss function gives more weight to hard examples and down-weights easy examples.
    It's particularly useful for imbalanced datasets.
    
    Parameters:
    -----------
    gamma : float, optional (default=2.0)
        Focusing parameter. Higher values give more weight to hard examples.
    alpha : float or list, optional (default=0.25)
        Class weight. Can be a single value or a list of weights for each class.
    reduction : str, optional (default='mean')
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions, shape (batch_size, num_classes)
        targets : torch.Tensor
            Ground truth labels, shape (batch_size, num_classes)
            
        Returns:
        --------
        loss : torch.Tensor
            Computed focal loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Probability of the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Apply focusing parameter
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # If alpha is a list/tensor with class-specific weights
                if isinstance(self.alpha, list):
                    self.alpha = torch.tensor(self.alpha).to(inputs.device)
                alpha_t = self.alpha.expand_as(targets) * targets + \
                         (1 - self.alpha).expand_as(targets) * (1 - targets)
            
            focal_weight = alpha_t * focal_weight
        
        # Compute the loss
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'sum'
            return loss.sum()

class ClassBalancedLoss(nn.Module):
    """
    Class Balanced Loss implementation for multi-label classification.
    
    This loss addresses class imbalance by re-weighting based on the effective number of samples.
    
    Parameters:
    -----------
    samples_per_class : list
        Number of samples for each class
    beta : float, optional (default=0.9999)
        Hyperparameter for computing effective number of samples
    gamma : float, optional (default=2.0)
        Focusing parameter for focal loss
    loss_type : str, optional (default='focal')
        Type of loss to use: 'focal' or 'bce'
    """
    
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, torch.tensor(samples_per_class).float())
        
        # Compute weights
        weights = (1.0 - self.beta) / effective_num
        
        # Normalize weights
        self.weights = weights / weights.sum() * len(samples_per_class)
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions, shape (batch_size, num_classes)
        targets : torch.Tensor
            Ground truth labels, shape (batch_size, num_classes)
            
        Returns:
        --------
        loss : torch.Tensor
            Computed class-balanced loss
        """
        weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'focal':
            # Compute focal loss
            probs = torch.sigmoid(inputs)
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * bce_loss
        else:
            # Compute BCE loss
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply class weights
        weights_batch = torch.matmul(targets, weights.unsqueeze(1)).squeeze(1)
        loss = loss * weights_batch.unsqueeze(1)
        
        return loss.mean()

# Example usage
if __name__ == "__main__":
    # Example inputs and targets
    inputs = torch.randn(4, 6)  # 4 samples, 6 classes
    targets = torch.tensor([
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1]
    ], dtype=torch.float32)
    
    # Example of using FocalLoss
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    loss_value = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss_value.item()}")
    
    # Example of using ClassBalancedLoss
    # First, compute samples per class
    samples_per_class = targets.sum(dim=0).tolist()
    print(f"Samples per class: {samples_per_class}")
    
    cb_loss = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.9999)
    loss_value = cb_loss(inputs, targets)
    print(f"Class Balanced Loss: {loss_value.item()}") 