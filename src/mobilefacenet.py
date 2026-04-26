import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    """Standard convolution block with BatchNorm and PReLU"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, 
                              groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    """Depthwise separable convolution without activation"""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, 
                              groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    """Depthwise separable convolution block"""
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.residual = residual
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    """Residual block with bottleneck"""
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, 
                                      padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet for face recognition
    Input: 112x112 RGB face image
    Output: 512-dim or 128-dim embedding
    """
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        self.embedding_size = embedding_size
        
        # Initial convolution: 112x112x3 -> 56x56x64
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        
        # Depthwise separable: 56x56x64 -> 56x56x64
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), 
                                   padding=(1, 1), groups=64)
        
        # Bottleneck blocks
        # Stage 1: 56x56x64 -> 28x28x64
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), 
                                  padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), 
                               stride=(1, 1), padding=(1, 1))
        
        # Stage 2: 28x28x64 -> 14x14x128
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), 
                                  padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), 
                               stride=(1, 1), padding=(1, 1))
        
        # Stage 3: 14x14x128 -> 7x7x128
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), 
                                  padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), 
                               stride=(1, 1), padding=(1, 1))
        
        # Final convolution: 7x7x128 -> 7x7x512
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Global Depthwise Convolution (replaces global pooling + FC)
        # 7x7x512 -> 1x1x512
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), 
                                      stride=(1, 1), padding=(0, 0))
        
        # Final linear projection to embedding
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, 112, 112)
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_size)
        """
        out = self.conv1(x)
        
        out = self.conv2_dw(out)
        
        out = self.conv_23(out)
        out = self.conv_3(out)
        
        out = self.conv_34(out)
        out = self.conv_4(out)
        
        out = self.conv_45(out)
        out = self.conv_5(out)
        
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        
        return out


def load_pretrained_mobilefacenet(embedding_size=512, pretrained_path=None):
    """
    Load MobileFaceNet with optional pretrained weights
    
    Args:
        embedding_size: Output embedding dimension (512 or 128)
        pretrained_path: Path to pretrained .pth file (optional)
    
    Returns:
        model: MobileFaceNet model
    """
    model = MobileFaceNet(embedding_size=embedding_size)
    
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel training)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded successfully!")
    else:
        print("Initialized with random weights")
    
    return model


# Test function
if __name__ == "__main__":
    # Test the model
    model = MobileFaceNet(embedding_size=512)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 112, 112)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with different embedding size
    model_128 = MobileFaceNet(embedding_size=128)
    output_128 = model_128(dummy_input)
    print(f"128-dim embedding output shape: {output_128.shape}")