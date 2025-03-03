import torch
import torch.nn as nn
from models.audio_lstm import AudioLSTM
from models.transformer import FeatureTransformer

class EnsembleModel(nn.Module):
    def __init__(self, audio_input_size, video_input_size, hidden_size, num_classes):
        super(EnsembleModel, self).__init__()
        self.audio_lstm = AudioLSTM(audio_input_size, hidden_size, 2, num_classes)
        self.video_transformer = FeatureTransformer(video_input_size, 8, 2, 2048)
        self.fc = nn.Linear(hidden_size + video_input_size, num_classes)

    def forward(self, audio_input, video_input):
        audio_out = self.audio_lstm(audio_input)
        video_out = self.video_transformer(video_input)
        combined = torch.cat((audio_out, video_out), dim=1)
        out = self.fc(combined)
        return out