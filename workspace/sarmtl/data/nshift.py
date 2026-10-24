"""Shifted-overlap search with the correlation normalised by the overlap energies (the critic's method), so a partial
overlap (e.g. 44 % of the image, at a 288 px shift) scores like a full one. Used by audit_split.py.

Thumbnails: the 64 x 64 whole-image thumbnails of extract_features.py, area-averaged to 32 x 32, minus their
Gaussian blur (sigma 1 px), zero-padded to 64 x 64. For train image A and a holdout image transformed by k (B):
  ncc(d) = sum_x A(x) B(x - d) / sqrt(sum_{x in overlap} A^2 * sum_{x in overlap} B^2), overlap >= 10 % of the image,
computed for all shifts d with FFTs. A[y, x] ~ B[y - dy, x - dx] in the same convention as near_dups.py. Candidates:
NCC >= 0.5 on >= 20 % of the image, or NCC >= 0.85 on >= 10 %. They are refined around the thumbnail shift by direct
NCC of the high-passed overlap: +-(4 + 2) px at 128 x 128 (stop if < 0.2), then +-(f + 1) px at full resolution
(f = side / 128).
"""
import cv2
import numpy as np
import torch

import near_dups
from near_dups import dihedral, highpass, ncc

S = 32
MIN_FRAC = 0.1
PEAK = 0.5
GATES = ((0.5, 0.2), (0.85, 0.1))  # candidate if (thumbnail NCC, overlap fraction) passes either gate
PRECHECK = 0.2  # high-passed NCC at 128 x 128 needed before the full-resolution refinement


def hp32(t64, size=S, center=False):
    """[N, 64, 64] uint8 -> [N, size, size] float32 high-passed (area average, minus Gaussian blur sigma size/32;
    center=True also subtracts the mean, as in the critic's spec; near_dups.py stage 6 used center=False)."""
    out = np.empty((len(t64), size, size), np.float32)
    for n, t in enumerate(t64.astype(np.float32)):
        a = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) if size != t.shape[0] else t
        h = a - cv2.GaussianBlur(a, (0, 0), size / 32)
        out[n] = h - h.mean() if center else h
    return out


class Search:
    """Precomputed spectra: FA (images as A), EA (overlap energy of A for every shift), FM (the mask)."""

    def __init__(self, hp, min_frac=MIN_FRAC, peak=PEAK):
        self.S = hp.shape[1]
        self.hp, self.P, self.min_frac, self.peak = hp, 2 * self.S, min_frac, peak
        S = self.S
        X = torch.from_numpy(hp)
        self.FA = torch.fft.rfft2(X, s=(self.P, self.P))
        m = torch.ones(S, S)
        self.FMc = torch.fft.rfft2(m, s=(self.P, self.P)).conj()
        self.EA = torch.fft.irfft2(torch.fft.rfft2(X * X, s=(self.P, self.P)) * self.FMc[None], s=(self.P, self.P))
        area = torch.fft.irfft2(torch.fft.rfft2(m, s=(self.P, self.P)) * self.FMc, s=(self.P, self.P))
        self.valid = area >= min_frac * S * S - 0.5
        self.area = area

    def partners(self, i, js):
        """Best (k, ncc, dx, dy, overlap fraction) per j in js for holdout image i under the 8 transforms."""
        best = {}
        S, PEAK = self.S, self.peak
        FAj = self.FA[js]
        EAj = self.EA[js]
        for k in range(8):
            B = torch.from_numpy(np.ascontiguousarray(dihedral(self.hp[i], k)))
            FBc = torch.fft.rfft2(B, s=(self.P, self.P)).conj()
            eb = torch.fft.irfft2(torch.fft.rfft2(torch.ones(S, S), s=(self.P, self.P)) *
                                  torch.fft.rfft2(B * B, s=(self.P, self.P)).conj(), s=(self.P, self.P))
            num = torch.fft.irfft2(FAj * FBc[None], s=(self.P, self.P))
            v = num / torch.sqrt((EAj * eb[None]).clamp_min(1e-6))
            v = torch.where(self.valid[None], v, torch.full_like(v, -2.0)).reshape(len(js), -1)
            peak, arg = v.max(1)
            for j, p, a in zip(np.asarray(js)[(peak >= PEAK).numpy()], peak[peak >= PEAK].tolist(),
                               arg[peak >= PEAK].tolist()):
                dy, dx = divmod(int(a), self.P)
                dy, dx = (dy - self.P if dy >= S else dy), (dx - self.P if dx >= S else dx)
                fr = float(self.area.reshape(-1)[int(a)]) / (S * S)
                if int(j) not in best or p > best[int(j)][1]:
                    best[int(j)] = (k, float(p), dx, dy, fr)
        return best


def _overlap(A, B, dx, dy):
    H, W = A.shape
    xa0, xa1, ya0, ya1 = max(0, dx), min(W, W + dx), max(0, dy), min(H, H + dy)
    if xa1 - xa0 < 4 or ya1 - ya0 < 4:
        return None, None
    return A[ya0:ya1, xa0:xa1], B[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]


def _best_shift(A, B, cx, cy, r):
    best = None
    for dy in range(cy - r, cy + r + 1):
        for dx in range(cx - r, cx + r + 1):
            pa, pb = _overlap(A, B, dx, dy)
            if pa is None or pa.size < 0.04 * A.size:
                continue
            v = ncc(pa, pb)
            if best is None or v > best[0]:
                best = (v, dx, dy)
    return best


def refine_verify(job):
    """job = (a, b, k, dx_t, dy_t): A = image a, B = image b transformed by k, thumbnail shift (dx_t, dy_t) ->
    (a, b, k, dx, dy, overlap fraction, NCC, high-pass NCC, equal-pixel fraction, mean abs. difference) at full
    resolution, or None."""
    a, b, k, dx_t, dy_t = job[:5]
    ts = job[5] if len(job) > 5 else S  # thumbnail size of the search that produced (dx_t, dy_t)
    A, B = near_dups.load(a), np.ascontiguousarray(dihedral(near_dups.load(b), k))
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        return None
    n = A.shape[0]
    f = max(n // 128, 1)
    a128 = cv2.resize(A, (n // f, n // f), interpolation=cv2.INTER_AREA)
    b128 = cv2.resize(B, (n // f, n // f), interpolation=cv2.INTER_AREA)
    h128a, h128b = a128 - cv2.GaussianBlur(a128, (0, 0), 1.5), b128 - cv2.GaussianBlur(b128, (0, 0), 1.5)
    scale = (n // f) / ts  # thumbnail px -> 128-res px
    r1 = int(np.ceil(scale)) + 2
    b1 = _best_shift(h128a, h128b, int(round(dx_t * scale)), int(round(dy_t * scale)), r1)
    if b1 is None:
        return None
    if b1[0] < PRECHECK:  # no structure agreement at 128 px: not the same scene, skip the full-resolution search
        return (a, b, k, b1[1] * f, b1[2] * f, None, None, None, None, None, b1[0])
    ha, hb = highpass(A), highpass(B)
    b2 = _best_shift(ha, hb, b1[1] * f, b1[2] * f, f + 1)
    if b2 is None:
        return None
    _, dx, dy = b2
    pa, pb = _overlap(A, B, dx, dy)
    return (a, b, k, dx, dy, pa.size / A.size, ncc(pa, pb), ncc(highpass(pa), highpass(pb)),
            float((pa == pb).mean()), float(np.abs(pa - pb).mean()), b1[0])
