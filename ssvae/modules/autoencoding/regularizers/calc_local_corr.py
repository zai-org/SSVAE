import torch


def calculate_pearson_autocorr(x, renorm=True):
    # v: (B, N, D)
    # return: scalar p
    # treat v as a batch B of N random variables, each has D observations
    D = x.size(-1)
    if renorm:
        x = x - x.mean(dim=-1, keepdim=True)
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        x = x / norm
    sim = torch.matmul(x, x.transpose(1, 2))  # (B, N, N)
    N = x.size(1)
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1)
    mean_corr = sim[:, mask == 1]
    mean_corr = mean_corr.mean()
    if not renorm:
        mean_corr = mean_corr / D
    return mean_corr


def transform_to_spatial_windows(t, window_sizes):
    B, D, F, H, W = t.shape
    _, hw, ww = window_sizes
    t = t.reshape(B, D, F, H//hw, hw, W//ww, ww).permute(0, 2, 3, 5, 4, 6, 1) # (B, F, H//hw, W//ww, hw, ww, D)
    t = t.reshape(B, F, H//hw*W//ww, hw*ww, D)
    return t


def merge_dimensions(t, window_sizes):
    _, hw, ww = window_sizes
    D = t.size(-1)
    t = t.reshape(-1, hw*ww, D)
    return t


def transform_to_spatial_temporal_windows(t, window_sizes):
    B, F, SWN, SN, D = t.size()
    tw, _, _ = window_sizes
    t = t.reshape(B, F//tw, tw, SWN, SN, D).permute(0, 1, 3, 2, 4, 5) # (B, FWN, SWN, tw, SN, D)
    t = t.reshape(B*(F//tw)*SWN, tw*SN, D)
    return t


def windowed_localcorr_z(latents, window_sizes, weight_type="average", renorm=True):
    '''
    mean, logvar: (B, D, F, H, W)
    window_size: List[int]: [temporal window, height window, width window]
    return: averaged local correlation across temporal/spatial, mean/logvar, windows, and batches
    '''
    z = latents
    B, D, F, H, W = latents.shape
    tw, hw, ww = window_sizes

    assert H % hw == 0 and W % ww == 0, f"H and W should be divisible by {hw} and {ww}"
    if F > 1:
        assert (F-1) % tw == 0, f"(F-1) should be divisible by {tw} for video windows"

    # transform into spatial windows
    latents = transform_to_spatial_windows(latents, window_sizes)
    # Calculate spatial average localcorrelation for images or the first frame of videos
    first_frame_or_image = latents[:, :1]
    first_frame_or_image = merge_dimensions(first_frame_or_image, window_sizes)
    first_frame_or_image_localcorr = calculate_pearson_autocorr(first_frame_or_image, renorm)
    # Calculate temporal-spatial average autocorrelation for videos
    if F > 1:
        video = latents[:, 1:] # (B, F, spatial_window_num, spatial_N, D)
        video = transform_to_spatial_temporal_windows(video, window_sizes)
        video_localcorr = calculate_pearson_autocorr(video, renorm)
        if weight_type == 'average':
            avg_localcorr = (first_frame_or_image_localcorr + video_localcorr) / 2
        elif weight_type == 'frame_weight':
            avg_localcorr = (first_frame_or_image_localcorr + (F-1) * video_localcorr) / F
        elif weight_type == 'window_weight':
            avg_localcorr = (first_frame_or_image_localcorr + (F-1)/tw * video_localcorr) / (1+(F-1)/tw)
        else:
            avg_localcorr = first_frame_or_image_localcorr * weight_type[0] + video_localcorr * weight_type[1]
    else:
        avg_localcorr = first_frame_or_image_localcorr
        video_localcorr = None

    return first_frame_or_image_localcorr, video_localcorr, avg_localcorr

