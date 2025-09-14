from model.layers import SeqToANNContainer, LIFSpike, add_dimention, FCLayer
import torch
import torch.nn as nn

class Layer1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(Layer1D, self).__init__()
        self.conv = SeqToANNContainer(nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding))
        self.bn = SeqToANNContainer(nn.BatchNorm1d(out_planes))
        self.act = LIFSpike()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class VGG9SNN(nn.Module):
    def __init__(self, num_cls, time_step, in_channels=5, input_size=36000):
        super(VGG9SNN, self).__init__()
        # 1D pooling
        pool = SeqToANNContainer(nn.AvgPool1d(2))

        self.num_cls = num_cls
        self.T = time_step
        self.in_channels = in_channels
        self.input_size = input_size
        self.act = LIFSpike()
        self.n_maps = 128
        
        # Stem layer for 1D input (wrapped for SNN)
        self.stem_conv = nn.Conv1d(self.in_channels, 64, 3, 1, 1)
        #self.stem_bn = nn.BatchNorm1d(64)

        # 1D Feature extraction layers
        self.features = nn.Sequential(
            Layer1D(64, 64, 3, 1, 1),
            pool,  # input_size / 2
            Layer1D(64, 128, 3, 1, 1),
            Layer1D(128, 128, 3, 1, 1),
            pool,  # input_size / 4
            Layer1D(128, 256, 3, 1, 1),
            Layer1D(256, 256, 3, 1, 1),
            pool,  # input_size / 8
        )
        
        # Calculate final sequence length after pooling
        # After 3 pooling layers: input_size / 8
        final_length = int(self.input_size / 8)
        
        self.shared_fc = nn.Sequential(
            SeqToANNContainer(nn.Flatten()),
            SeqToANNContainer(nn.Linear(256 * final_length, 512)),  # 여기서 final_length 사용
            SeqToANNContainer(nn.ReLU()),
            SeqToANNContainer(nn.Dropout(0.5))
        )
        
        # Classifier
        self.classifier_han_yeol = SeqToANNContainer(nn.Linear(512, self.num_cls))
        self.classifier_huh_sil = SeqToANNContainer(nn.Linear(512, self.num_cls))
        
        self.regressor_han = SeqToANNContainer(nn.Linear(512, 1))
        self.regressor_yeol = SeqToANNContainer(nn.Linear(512, 1))
        self.regressor_huh = SeqToANNContainer(nn.Linear(512, 1))
        self.regressor_sil = SeqToANNContainer(nn.Linear(512, 1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inp_1):
        # inp_1 shape: [B, C, L] for 1D data where C=5, L=36000
        
        # direct learning
        stem_inp= self.stem_conv(inp_1)
        
        # Add time dimension first: [B, C, L] -> [T, B, C, L]
        if len(inp_1.shape) == 3:  # [B, C, L]
            spike_inp = add_dimention(stem_inp, self.T)
        else: 
            spike_inp = stem_inp
        
        # Feature extraction
        x = self.features(spike_inp)
        
        # Classification
        x = self.shared_fc(x)
        x1 = self.classifier_han_yeol(x)
        x2 = self.classifier_huh_sil(x)
        x3 = self.regressor_han(x)
        x4 = self.regressor_yeol(x)
        x5 = self.regressor_huh(x)
        x6 = self.regressor_sil(x)
        #print(x.shape)
        return x1.mean(0), x2.mean(0), x3.mean(0), x4.mean(0), x5.mean(0), x6.mean(0)


# # 사용 예시
# if __name__ == "__main__":
#     # 테스트 데이터 생성
#     batch_size = 4
#     channels = 5
#     sequence_length = 36000
#     num_classes = 2
#     time_steps = 4
    
#     # 1D 입력 데이터: [B, C, L]
#     test_input = torch.randn(batch_size, channels, sequence_length).cuda()
    
#     print("=== VGG9SNN (Full Version) ===")
#     model = VGG9SNN(num_cls=num_classes, time_step=time_steps, 
#                     in_channels=channels, input_size=sequence_length).cuda()
    
#     print(f"Input shape: {test_input.shape}")
    
#     try:
#         output = model(test_input)
#         print(f"Output shape: {output.shape}")
        
#         # 파라미터 수 계산
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"Total parameters: {total_params:,}")
        
#     except Exception as e:
#         print(f"Error in full model: {e}")

