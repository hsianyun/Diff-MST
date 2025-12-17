import numpy as np
from laion_clap import CLAP_Module
import torchaudio

def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt("/home/hsianyun/Diff-MST/music_audioset_epoch_15_esc_90.14.pt")

text = "the audio sounds extremely bright"
text_embedding = model.get_text_embedding([text])[0]

audio_path_mix = "/home/hsianyun/Diff-MST/audio/orig_track3.wav"
waveform_mix, sr_mix = torchaudio.load(audio_path_mix)
audio_embedding_mix = model.get_audio_embedding_from_data(waveform_mix, use_tensor = True)[0].detach().cpu().numpy()

audio_path_bright = "/home/hsianyun/Diff-MST/audio/bright_track3.wav"
waveform_bright, sr_bright = torchaudio.load(audio_path_bright)
audio_embedding_bright = model.get_audio_embedding_from_data(waveform_bright, use_tensor = True)[0].detach().cpu().numpy()

similarity_mix = cosine_similarity(text_embedding, audio_embedding_mix)
similarity_bright = cosine_similarity(text_embedding, audio_embedding_bright)

similarity_audios = cosine_similarity(audio_embedding_mix, audio_embedding_bright)

print(f"Cosine similarity between text and mixed audio: {similarity_mix}")
print(f"Cosine similarity between text and bright audio: {similarity_bright}")
print(f"Cosine similarity between mixed audio and bright audio: {similarity_audios}")