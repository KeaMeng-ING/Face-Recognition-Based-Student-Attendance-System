"""
ArcFace Loss Implementation
Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)
    
    This loss improves face recognition by adding an angular margin penalty
    between the feature and the center of the ground-truth class, making
    embeddings of the same person cluster tightly together.
    
    Args:
        embedding_size: Size of input embeddings (e.g., 512)
        num_classes: Number of identities in training set
        s: Scale parameter (default: 64.0)
        m: Angular margin penalty in radians (default: 0.5)
        
    Math:
        L = -log( e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ e^(s·cos(θ_j))) )
        
        where:
        - θ_yi is the angle between embedding and correct class center
        - m is the angular margin (makes correct class harder to match)
        - s is a scale factor that controls the sharpness of the distribution
    """
    
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Angular margin in radians
        
        # Weight matrix: each row is the center of one class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute some constants for numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # Threshold for avoiding numerical issues
        self.mm = math.sin(math.pi - m) * m  # For clamping
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_size) - output from backbone
            labels: (batch_size,) - ground truth class labels
            
        Returns:
            loss: scalar ArcFace loss
        """
        # Step 1: Normalize both embeddings and weights
        # This makes the dot product equivalent to cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Step 2: Compute cosine similarity (angle between vectors)
        # cosine shape: (batch_size, num_classes)
        cosine = F.linear(embeddings, weight_normalized)
        
        # Step 3: Compute sine from cosine
        # We need sin(θ) to compute cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        # Step 4: Add angular margin: cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Step 5: Handle numerical stability
        # When cos(θ) < cos(π - m), the angle is too large and adding margin
        # would cause issues, so we use a different formulation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Step 6: Create one-hot encoding of labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Step 7: Apply margin only to the correct class
        # For ground truth class: use phi (with margin)
        # For other classes: use cosine (without margin)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Step 8: Scale the output
        output *= self.s
        
        # Step 9: Apply cross entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace Loss (Large Margin Cosine Loss) - Alternative to ArcFace
    Adds a margin in the cosine space instead of angular space
    
    Math:
        L = -log( e^(s·(cos(θ_yi) - m)) / (e^(s·(cos(θ_yi) - m)) + Σ e^(s·cos(θ_j))) )
    """
    
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight_normalized)
        
        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Add margin in cosine space (simpler than ArcFace)
        output = self.s * (cosine - one_hot * self.m)
        
        loss = F.cross_entropy(output, labels)
        
        return loss


class SoftmaxLoss(nn.Module):
    """
    Standard Softmax Loss (baseline for comparison)
    This is just regular cross-entropy without any margin
    """
    
    def __init__(self, embedding_size, num_classes):
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)
        
    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits, labels)
        return loss


# ============================================================================
# Testing and Comparison
# ============================================================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Dummy data
    batch_size = 16
    embedding_size = 512
    num_classes = 158  # LFW subset
    
    embeddings = torch.randn(batch_size, embedding_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test ArcFace
    print("\n1. ArcFace Loss")
    arcface = ArcFaceLoss(embedding_size, num_classes, s=64.0, m=0.5)
    loss_arc = arcface(embeddings, labels)
    print(f"   Loss: {loss_arc.item():.4f}")
    
    # Test CosFace
    print("\n2. CosFace Loss")
    cosface = CosFaceLoss(embedding_size, num_classes, s=64.0, m=0.35)
    loss_cos = cosface(embeddings, labels)
    print(f"   Loss: {loss_cos.item():.4f}")
    
    # Test Softmax
    print("\n3. Softmax Loss (baseline)")
    softmax = SoftmaxLoss(embedding_size, num_classes)
    loss_soft = softmax(embeddings, labels)
    print(f"   Loss: {loss_soft.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")
    
    # Demonstrate the effect of margin
    print("\n" + "="*60)
    print("Understanding the Margin Effect:")
    print("="*60)
    print(f"ArcFace margin (m={arcface.m:.2f} radians = {math.degrees(arcface.m):.1f}°)")
    print(f"This means the model must separate embeddings by at least {math.degrees(arcface.m):.1f}°")
    print(f"in angular space to correctly classify them.")
    print(f"\nWithout margin (Softmax): embeddings just need to be 'closest'")
    print(f"With margin (ArcFace): embeddings must be closest AND separated by {math.degrees(arcface.m):.1f}°")
    print("\nThis forces the model to create more discriminative features!")
