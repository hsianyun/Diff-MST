import torch
import torchaudio
import numpy as np
import gradio as gr
import pyloudnorm as pyln
from mst.utils import load_diffmst, run_diffmst

meter = pyln.Meter(44100)

method = {
    "model": load_diffmst(
        config_path="./test/naive.yaml",
        ckpt_path="/mnt/gestalt/home/rakec/output/diff-mst/epoch=247-step=77624.ckpt",
    ),
    "func": run_diffmst
}

def slider_max_update(filepath):
    if filepath is None or filepath == "":
        return gr.update(maximum=0)
    
    info = torchaudio.info(filepath)
    duration = info.num_frames / info.sample_rate
    return gr.update(maximum=duration - 10)

def process_input(filepath):
    audio, sr = torchaudio.load(filepath, backend="soundfile")

    # resample to 44.1kHz if needed
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        audio = resampler(audio)
        sr = 44100
    
    return audio, sr

def audio_control(
    raw_file_1, raw_file_2, raw_file_3, raw_file_4,
    raw_file_5, raw_file_6, raw_file_7, raw_file_8,
    ref_file, ref_start_time
):
    if ref_file is None:
        raise ValueError("Reference file is required for audio control.")
    
    ref_audio, ref_sr = process_input(ref_file)
    assert ref_sr == 44100, "All audio files must have a sample rate of 44.1kHz."
    ref_audio = ref_audio.unsqueeze(0)

    raw_files = [
        raw_file_1, raw_file_2, raw_file_3, raw_file_4,
        raw_file_5, raw_file_6, raw_file_7, raw_file_8
    ]

    raw_tracks = []
    tracks_idx = []
    lengths = []
    for i, file in enumerate(raw_files):
        if file is not None:
            audio, sr = process_input(file)
            assert sr == 44100, "All audio files must have a sample rate of 44.1kHz."

            chs, seq_len = audio.shape
            for ch_idx in range(chs):
                raw_tracks.append(audio[ch_idx : ch_idx + 1, :])
                if i not in tracks_idx:
                    tracks_idx.append(i)
                lengths.append(seq_len)

    if len(raw_tracks) == 0:
        raise ValueError("At least one raw track must be provided.")
    
    max_length = max(lengths)
    for track_idx in range(len(raw_tracks)):
        raw_tracks[track_idx] = torch.nn.functional.pad(
            raw_tracks[track_idx], (0, max_length - lengths[track_idx])
        )
    
    raw_tracks = torch.cat(raw_tracks, dim=0)
    raw_tracks = raw_tracks.view(1, -1, max_length)

    start_idx = int(ref_start_time * 44100)
    ref_segment = ref_audio[..., start_idx: start_idx + 44100 * 10]
    
    model, mix_console = method["model"]
    model = model.to("cpu")
    mix_console = mix_console.to("cpu")
    func = method["func"]

    with torch.no_grad():
        result = func(
            raw_tracks.clone(),
            ref_segment.clone(),
            model,
            mix_console,
            track_start_idx=0,
            ref_start_idx=start_idx,
        )

        (
            pred_mix,
            pred_mixed_tracks,
            pred_track_param_dict,
            pred_fx_bus_param_dict,
            pred_master_bus_param_dict,
        ) = result
        
        bs, chs, seq_len = pred_mix.shape

        mix_lufs_db = meter.integrated_loudness(
            pred_mix.squeeze(0).permute(1, 0).numpy()
        )
        lufs_delta_db = -16.0 - mix_lufs_db
        pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

        # tracks
        pred_tracks = []
        for i in range(8):
            if i not in tracks_idx:
                pred_tracks.append(None)
            else:
                pred_tracks.append(pred_mixed_tracks[:, :, tracks_idx.index(i), :])

        pred_track_1 = pred_tracks[0] if pred_tracks[0] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_2 = pred_tracks[1] if pred_tracks[1] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_3 = pred_tracks[2] if pred_tracks[2] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_4 = pred_tracks[3] if pred_tracks[3] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_5 = pred_tracks[4] if pred_tracks[4] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_6 = pred_tracks[5] if pred_tracks[5] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_7 = pred_tracks[6] if pred_tracks[6] is not None else torch.zeros(bs, chs, seq_len)
        pred_track_8 = pred_tracks[7] if pred_tracks[7] is not None else torch.zeros(bs, chs, seq_len)

    # convert from float32 tensor to numpy array
    pred_mix = pred_mix.squeeze(0).T.numpy()
    pred_mix = (pred_mix * 32767).clip(-32768, 32767).astype(np.int16)

    # tracks
    pred_track_1 = pred_track_1.squeeze(0).T.numpy()
    pred_track_2 = pred_track_2.squeeze(0).T.numpy()
    pred_track_3 = pred_track_3.squeeze(0).T.numpy()
    pred_track_4 = pred_track_4.squeeze(0).T.numpy()
    pred_track_5 = pred_track_5.squeeze(0).T.numpy()
    pred_track_6 = pred_track_6.squeeze(0).T.numpy()
    pred_track_7 = pred_track_7.squeeze(0).T.numpy()
    pred_track_8 = pred_track_8.squeeze(0).T.numpy()
    pred_track_1 = (pred_track_1 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_2 = (pred_track_2 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_3 = (pred_track_3 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_4 = (pred_track_4 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_5 = (pred_track_5 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_6 = (pred_track_6 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_7 = (pred_track_7 * 32767).clip(-32768, 32767).astype(np.int16)
    pred_track_8 = (pred_track_8 * 32767).clip(-32768, 32767).astype(np.int16)

    return (44100, pred_mix), (44100, pred_track_1), (44100, pred_track_2), (44100, pred_track_3), (44100, pred_track_4), (44100, pred_track_5), (44100, pred_track_6), (44100, pred_track_7), (44100, pred_track_8)