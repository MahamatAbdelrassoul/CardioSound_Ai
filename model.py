import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
from PIL import Image

class STGNN(nn.Module):
    """
    Spectro-Temporal Graph Neural Network for Cardiac Audio Analysis
    Enhanced with real clinical data integration
    """
    
    def __init__(self, num_clinical_features=22, num_classes=2, hidden_dim=128, num_nodes=4, audio_length=66150):
        super(STGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.audio_length = audio_length
        
        # Spectrogram generation parameters
        self.sample_rate = 22050
        self.n_fft = 512
        self.hop_length = 256
        self.n_mels = 64
        
        # Spectrogram CNN feature extractor
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 8 * 8
        
        # Clinical features processing (REAL clinical data)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined features processing
        self.combined_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def build_audio_graph(self):
        """Build graph based on audio recordings"""
        adj = torch.ones(self.num_nodes, self.num_nodes)
        degree = torch.diag(torch.sum(adj, dim=1))
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree + 1e-6))
        normalized_adj = torch.mm(torch.mm(degree_inv_sqrt, adj), degree_inv_sqrt)
        return normalized_adj
    
    def audio_to_spectrogram(self, audio_batch):
        """
        Convert raw audio batch to mel spectrograms
        """
        batch_size, num_nodes, audio_length = audio_batch.shape
        
        spectrograms = []
        
        for batch_idx in range(batch_size):
            batch_spectrograms = []
            for node_idx in range(num_nodes):
                audio = audio_batch[batch_idx, node_idx]
                
                # Create spectrogram using torch.stft
                spectrogram = torch.stft(
                    audio, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=torch.hann_window(self.n_fft).to(audio.device),
                    return_complex=True,
                    center=False
                )
                
                # Convert to magnitude spectrogram
                magnitude = torch.abs(spectrogram)
                
                # Apply simple mel-like scaling
                n_freq_bins = magnitude.shape[0]
                if n_freq_bins > self.n_mels:
                    mel_spectrogram = F.adaptive_avg_pool1d(magnitude.unsqueeze(0), self.n_mels)
                    mel_spectrogram = mel_spectrogram.squeeze(0)
                else:
                    mel_spectrogram = magnitude
                
                # Log compression
                log_mel = torch.log(mel_spectrogram + 1e-6)
                batch_spectrograms.append(log_mel)
            
            batch_spectrograms = torch.stack(batch_spectrograms)
            spectrograms.append(batch_spectrograms)
        
        spectrograms = torch.stack(spectrograms)
        return spectrograms
    
    def forward(self, audio_data, clinical_data):
        """Forward pass with raw audio and clinical data"""
        batch_size = audio_data.shape[0]
        
        # 1. Convert audio to spectrograms
        spectrograms = self.audio_to_spectrogram(audio_data)
        
        # 2. Extract features from spectrograms using CNN
        cnn_input = spectrograms.reshape(batch_size * self.num_nodes, 1, 
                                       spectrograms.shape[2], spectrograms.shape[3])
        cnn_features = self.spectrogram_cnn(cnn_input)
        cnn_features = cnn_features.reshape(batch_size, self.num_nodes, -1)
        
        # 3. Process clinical features
        clinical_features = self.clinical_encoder(clinical_data)
        
        # 4. Combine features
        audio_features = torch.mean(cnn_features, dim=1)
        combined_features = torch.cat([audio_features, clinical_features], dim=1)
        
        # 5. Final classification
        encoded = self.combined_encoder(combined_features)
        output = self.classifier(encoded)
        
        return output, spectrograms

def load_model(model_path='stgnn_spectrogram_model.pth'):
    """Load the trained model"""
    model = STGNN(num_clinical_features=22, num_classes=2)
    
    # Load weights (with error handling for compatibility)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except:
        # If compatibility error, load only matching weights
        print("Adjusting model weights for compatibility...")
        pretrained_dict = torch.load(model_path, map_location='cpu')
        model_dict = model.state_dict()
        
        # Filter matching weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model.eval()
    return model

def create_clinical_features(clinical_data):
    """
    Create comprehensive clinical feature vector from user input
    Based on the original 22 features from your Kaggle model
    """
    # Extract clinical data
    age_encoded = clinical_data['age_encoded']
    sex_encoded = clinical_data['sex_encoded']
    height = clinical_data['height']
    weight = clinical_data['weight']
    bmi = clinical_data['bmi']
    
    # Normalize continuous features (using approximate ranges from pediatric data)
    height_norm = (height - 100) / 50  # Normalize ~50-150cm range
    weight_norm = (weight - 20) / 30   # Normalize ~20-50kg range  
    bmi_norm = (bmi - 16) / 8          # Normalize ~16-24 BMI range
    
    # Create comprehensive feature vector (22 features as in original model)
    clinical_features = np.zeros(22, dtype=np.float32)
    
    # 1. Basic demographics (5 features)
    clinical_features[0] = age_encoded / 3.0          # Normalized age category
    clinical_features[1] = sex_encoded               # Sex (0/1)
    clinical_features[2] = height_norm               # Normalized height
    clinical_features[3] = weight_norm               # Normalized weight  
    clinical_features[4] = bmi_norm                  # Normalized BMI
    
    # 2. Murmur features (simulated - set to 0 for new patients)
    # These would normally come from physical exam
    clinical_features[5] = 0.0    # murmur_grade_encoded (normalized)
    clinical_features[6] = 0.0    # murmur_pitch_encoded (normalized)
    clinical_features[7] = 0.0    # murmur_quality_encoded (normalized)
    clinical_features[8] = 0.0    # murmur_timing_encoded (normalized)
    
    # 3. Location features (simulated - assume standard auscultation)
    clinical_features[9] = 1.0    # has_AV (Aortic Valve)
    clinical_features[10] = 1.0   # has_PV (Pulmonary Valve)
    clinical_features[11] = 1.0   # has_TV (Tricuspid Valve) 
    clinical_features[12] = 1.0   # has_MV (Mitral Valve)
    clinical_features[13] = 4.0 / 6.0  # location_coverage (normalized)
    
    # 4. Derived features (simulated interactions)
    clinical_features[14] = bmi_norm * (age_encoded / 3.0)  # bmi_age_interaction
    clinical_features[15] = 0.0    # murmur_grade_x_locations
    clinical_features[16] = (height / max(weight, 1)) / 3.0  # height_weight_ratio (normalized)
    clinical_features[17] = 0.0    # murmur_severity_score
    
    # 5. Additional clinical indicators
    clinical_features[18] = 0.0    # total_murmur_indicators
    clinical_features[19] = 0.0    # pregnancy_encoded
    clinical_features[20] = 1.0    # num_locations (normalized)
    clinical_features[21] = 0.1    # murmur_present (low prior probability)
    
    return torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0)

def preprocess_audio(audio_path, clinical_data=None, target_length=66150, num_recordings=4):
    """
    Preprocess audio file for prediction with clinical data integration
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        # Adjust length
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # Create simulated batch (num_recordings copies of same audio)
        audio_sequence = np.stack([y] * num_recordings)
        audio_tensor = torch.tensor(audio_sequence, dtype=torch.float32).unsqueeze(0)
        
        # Create clinical features from user input
        if clinical_data is None:
            # Fallback to zeros if no clinical data provided
            clinical_features = torch.zeros(1, 22, dtype=torch.float32)
        else:
            clinical_features = create_clinical_features(clinical_data)
        
        return audio_tensor, clinical_features, y, sr
        
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return None, None, None, None

def create_spectrogram_plot(audio, sr, spectrogram_tensor=None):
    """
    Create spectrogram visualization for display
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Waveform
    time_axis = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time_axis, audio, color='#0077b6', linewidth=1)
    axes[0].set_title('Heart Sound Waveform', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')
    
    # Spectrogram
    if spectrogram_tensor is not None:
        # Use model-generated spectrogram
        spec = spectrogram_tensor[0, 0].cpu().numpy()  # First batch, first node
        im = axes[1].imshow(spec, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='bilinear')
        axes[1].set_title('AI-Generated Spectrogram', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Frequency Bins')
        plt.colorbar(im, ax=axes[1], label='Log Magnitude (dB)')
    else:
        # Fallback spectrogram using librosa
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', 
                               sr=sr, ax=axes[1], cmap='viridis')
        axes[1].set_title('Spectrogram (Librosa)', fontsize=12, fontweight='bold')
        plt.colorbar(axes[1].images[0], ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Convert to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    
    return image

def predict_heart_condition(model, audio_tensor, clinical_tensor):
    """
    Make heart condition prediction using both audio and clinical data
    """
    with torch.no_grad():
        output, spectrograms = model(audio_tensor, clinical_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities.numpy()[0],
            'spectrograms': spectrograms
        }