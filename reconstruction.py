import argparse
import os
import torch
import numpy as np
import imageio
from omegaconf import OmegaConf
from ssvae.util import instantiate_from_config
from ssvae.data.video_data_wds import read_video


def load_video(path, num_frames=17, size=256):
    vframes, _, _ = read_video(path, pts_unit="sec")
    total = vframes.shape[0]
    if total == 0:
        raise RuntimeError(f"Failed to read video {path}")
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    vframes = vframes[idxs]  # [T, H, W, C]
    vframes = vframes.permute(0, 3, 1, 2)  # [T, 3, H, W]
    # resize
    vframes = torch.stack(
        [
            torch.nn.functional.interpolate(
                f.unsqueeze(0).float(),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[0]
            for f in vframes
        ]
    )
    vframes = vframes / 127.5 - 1.0
    vframes = vframes.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
    return vframes


def save_video(frames, path, fps=16):
    # frames: [T, 3, H, W], float32 [-1,1]
    frames = frames.cpu().detach()
    frames = ((frames + 1) * 127.5).clamp(0, 255).byte().numpy()  # [T, 3, H, W]
    frames = np.transpose(frames, [0, 2, 3, 1])  # [T, H, W, 3]
    with imageio.get_writer(path, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save reconstructed video"
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    assert os.path.exists(args.config), f"Config {args.config} not found"
    assert os.path.exists(args.input), f"Input video {args.input} not found"

    input_basename = os.path.basename(args.input)
    output_path = os.path.join(args.output, input_basename)
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    print("Instantiating model...")
    model = instantiate_from_config(config.model)
    model = model.to(args.device).eval().to(torch.float16)

    print(f"Loading video from {args.input} ...")
    video = load_video(args.input, num_frames=args.num_frames, size=args.image_size)
    video = video.to(args.device).to(torch.float16)

    with torch.no_grad():
        latents = model.encode(video)
        recon = model.decode(latents)
    # recon: [1, 3, T, H, W]
    recon = recon[0].permute(1, 0, 2, 3)  # [T, 3, H, W]
    print(f"Saving reconstructed video to {output_path}")
    save_video(recon, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
