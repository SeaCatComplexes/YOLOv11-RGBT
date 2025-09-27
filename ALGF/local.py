import torch
import torch.nn as nn
import torch.nn.functional as F


class LAFM(nn.Module):
    def __init__(self, in_channels):
        super(LAFM, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        max_pool = F.max_pool2d(x, kernel_size=x.size()[2:])
        
        # Concatenate the results of average and max pooling
        pool_out = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply 1x1 convolution to reduce the number of channels
        conv1x1_out = self.relu(self.conv1x1(pool_out))
        
        # Apply another round of pooling and concatenation
        avg_pool_2 = F.avg_pool2d(conv1x1_out, kernel_size=conv1x1_out.size()[2:])
        max_pool_2 = F.max_pool2d(conv1x1_out, kernel_size=conv1x1_out.size()[2:])
        pool_out_2 = torch.cat([avg_pool_2, max_pool_2], dim=1)
        
        # Apply 3x3 convolution twice to further process the features
        conv3x3_1_out = self.relu(self.conv3x3_1(pool_out_2))
        conv3x3_2_out = self.relu(self.conv3x3_2(conv3x3_1_out))
        
        # Upsample the output to match the original input size
        upsample_out = F.interpolate(conv3x3_2_out, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        # Apply sigmoid activation to get the attention map
        attention_map = self.sigmoid(upsample_out)
        
        return attention_map
    
class LLModule(nn.Module):
    def __init__(self, in_channels):
        super(LLModule, self).__init__()
        self.lafm1 = LAFM(in_channels)
        self.lafm2 = LAFM(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # Generate attention maps for both inputs using LAFM
        attention_map1 = self.lafm1(x1)
        attention_map2 = self.lafm2(x2)
        
        # Multiply the attention maps with the original inputs
        attended_x1 = x1 * attention_map1
        attended_x2 = x2 * attention_map2
        
        # Concatenate the attended features
        concat_features = torch.cat([attended_x1, attended_x2], dim=1)
        
        # Apply 1x1 convolution to reduce the number of channels
        conv1x1_out = self.relu(self.conv1x1(concat_features))
        
        # Apply 3x3 convolution to further process the features
        final_out = self.relu(self.conv3x3(conv1x1_out))
        
        return final_out

# Example usage
if __name__ == "__main__":
    # Assuming input images are of shape [batch_size, channels, height, width]
    batch_size, channels, height, width = 4, 64, 256, 256
    
    # Create dummy input tensors for PET and CT images
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Instantiate the LLModule
    ll_module = LLModule(channels)
    
    # Forward pass through the LLModule
    output = ll_module(x1, x2)
    
    print("Output shape:", output.shape)