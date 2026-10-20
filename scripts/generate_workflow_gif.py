#!/usr/bin/env python3
"""
Generate the SecureLoRA "Proposed End-to-End Workflow" animated GIF.

Output : docs/assets/securelora_workflow.gif   (1920x1080, 16:9, ~15 s, loops)
Usage  : python scripts/generate_workflow_gif.py [--preview T [T ...]] [--out PATH]

Colour semantics (consistent throughout):
    green  = check passed          amber = conditional / evidence unavailable
    red    = rejected outcome      cyan  = active data flow / current stage
The animation depicts the *proposed* workflow only; it contains no benchmark
numbers and makes no hardware-backed (TPM / enclave) claims.
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Canvas / timing
# --------------------------------------------------------------------------- #
W, H = 1920, 1080
S = 2                       # supersampling factor (render 2x, downsample)
FRAME_MS = 80               # 12.5 fps (GIF delay is in 1/100 s -> 8 cs exactly)
N_FRAMES = 188              # 15.04 s loop
T_FADE0, T_FADE1 = 14.2, 14.96

# Stage activation times (seconds)
T0 = {1: 0.4, 2: 1.8, 3: 3.4, 4: 5.0, 5: 7.0, 6: 8.4, 7: 10.0, 8: 11.6}
T_FULL = 12.8               # all stages visible simultaneously from here

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG = (9, 15, 30)
PANEL = (16, 26, 46)
PANEL2 = (23, 37, 64)
BORDER = (46, 62, 95)
BORDER2 = (70, 90, 130)
TXT = (228, 234, 246)
DIM = (128, 143, 172)
TITLE_DIM = (150, 163, 190)
ICON_DIM = (78, 96, 132)
BAR = (84, 102, 140)
CYAN = (56, 189, 248)
PURPLE = (167, 139, 250)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)
RED = (248, 113, 113)
NAVY = (8, 14, 28)

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
MW, MH = 350, 330
XS = [80, 550, 1020, 1490]
Y1, Y2 = 110, 520

TITLES = {
    1: ("Dataset Ingestion", "& Validation"),
    2: ("PII Detection", "& Masking"),
    3: ("LoRA / Optional", "DP-LoRA Training"),
    4: ("Adapter Screening", ""),
    5: ("Integrity Verification", "& Signing"),
    6: ("Device-Bound", "Encryption"),
    7: ("Deployment", "Verification"),
    8: ("Authorized Loading &", "Inference Validation"),
}


def pos(i):
    """Top-left corner of stage i's module (snake layout)."""
    if i <= 4:
        return XS[i - 1], Y1
    return XS[8 - i], Y2


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def ease(u):
    u = clamp(u)
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    u = clamp(u)
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


def fade(color, a, behind):
    return lerp(behind, color, a)


def P(t, i, off, dur=0.3):
    """Eased 0..1 progress of an element of stage i that starts at T0[i]+off."""
    return ease((t - (T0[i] + off)) / dur)


def fade_factor(t):
    return 1.0 - ease((t - T_FADE0) / (T_FADE1 - T_FADE0))


def R(t, F, i, off, dur=0.3):
    return P(t, i, off, dur) * F


# --------------------------------------------------------------------------- #
# Drawing wrapper (all coordinates are logical 1920x1080 units)
# --------------------------------------------------------------------------- #
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_FONTS = {}


def font(bold, italic, size):
    key = (bold, italic, size)
    if key not in _FONTS:
        name = "DejaVuSans"
        if bold:
            name += "-Bold"
        elif italic:
            name += "-Oblique"
        _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), int(round(size * S)))
    return _FONTS[key]


class Cv:
    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img)

    def rrect(self, x, y, w, h, r=8, fill=None, outline=None, width=1):
        self.d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=r * S,
                                 fill=fill, outline=outline, width=max(1, int(round(width * S))))

    def rect(self, x, y, w, h, fill=None, outline=None, width=1):
        self.d.rectangle([x * S, y * S, (x + w) * S, (y + h) * S], fill=fill, outline=outline,
                         width=max(1, int(round(width * S))))

    def ellipse(self, cx, cy, rx, ry=None, fill=None, outline=None, width=1):
        ry = rx if ry is None else ry
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
                       fill=fill, outline=outline, width=max(1, int(round(width * S))))

    def arc(self, cx, cy, rx, ry, a0, a1, fill, width=2):
        self.d.arc([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], a0, a1,
                   fill=fill, width=max(1, int(round(width * S))))

    def line(self, pts, fill, width=2):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill,
                    width=max(1, int(round(width * S))), joint="curve")

    def poly(self, pts, fill):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def text(self, x, y, s, size=16, fill=TXT, bold=False, italic=False, anchor="lm",
             maxw=None, minsize=10):
        sz = size
        f = font(bold, italic, sz)
        if maxw:
            while sz > minsize and f.getlength(s) / S > maxw:
                sz -= 0.5
                f = font(bold, italic, sz)
        self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
        return f.getlength(s) / S

    def dashed_line(self, p0, p1, fill, width=2, dash=7, gap=5):
        (x0, y0), (x1, y1) = p0, p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        s = 0.0
        while s < length:
            e = min(s + dash, length)
            self.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill, width)
            s += dash + gap

    def dashed_rect(self, x, y, w, h, fill, width=2, dash=8, gap=5):
        self.dashed_line((x, y), (x + w, y), fill, width, dash, gap)
        self.dashed_line((x + w, y), (x + w, y + h), fill, width, dash, gap)
        self.dashed_line((x + w, y + h), (x, y + h), fill, width, dash, gap)
        self.dashed_line((x, y + h), (x, y), fill, width, dash, gap)

    def arrow(self, pts, fill, width=4, head=13):
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        bx, by = x1 - ux * head, y1 - uy * head
        self.line(list(pts[:-1]) + [(bx, by)], fill, width)
        px, py = -uy, ux
        self.poly([(x1, y1), (bx + px * head * 0.55, by + py * head * 0.55),
                   (bx - px * head * 0.55, by - py * head * 0.55)], fill)


def bezier(p0, p1, p2, p3, n=60):
    pts = []
    for k in range(n + 1):
        u = k / n
        a, b, c, d = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def path_point(pts, s):
    seg = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)]
    total = sum(seg)
    target = s * total
    acc = 0.0
    for k, ln in enumerate(seg):
        if acc + ln >= target:
            u = (target - acc) / ln if ln else 0
            return (pts[k][0] + (pts[k + 1][0] - pts[k][0]) * u,
                    pts[k][1] + (pts[k + 1][1] - pts[k][1]) * u)
        acc += ln
    return pts[-1]


# --------------------------------------------------------------------------- #
# Icons (drawn with primitives, centred on cx, cy)
# --------------------------------------------------------------------------- #
def icon_db(cv, cx, cy, c):
    cv.ellipse(cx, cy + 12, 16, 6, fill=PANEL, outline=c, width=2)
    cv.rect(cx - 16, cy - 12, 32, 24, fill=PANEL)
    cv.line([(cx - 16, cy - 12), (cx - 16, cy + 12)], c, 2)
    cv.line([(cx + 16, cy - 12), (cx + 16, cy + 12)], c, 2)
    cv.arc(cx, cy + 1, 16, 6, 0, 180, c, 2)
    cv.arc(cx, cy - 5, 16, 6, 0, 180, c, 2)
    cv.ellipse(cx, cy - 12, 16, 6, fill=PANEL, outline=c, width=2)


def icon_magnifier(cv, cx, cy, c):
    cv.ellipse(cx - 4, cy - 4, 12, fill=None, outline=c, width=3)
    cv.line([(cx + 5, cy + 5), (cx + 16, cy + 16)], c, 4)


def icon_network(cv, cx, cy, c):
    layers = [(-18, [-11, 0, 11]), (0, [-15, -5, 5, 15]), (18, [-11, 0, 11])]
    for li in range(2):
        for ya in layers[li][1]:
            for yb in layers[li + 1][1]:
                cv.line([(cx + layers[li][0], cy + ya), (cx + layers[li + 1][0], cy + yb)],
                        lerp(PANEL, c, 0.45), 1)
    for lx, ys in layers:
        for ya in ys:
            cv.ellipse(cx + lx, cy + ya, 4, fill=c)


def icon_shield(cv, cx, cy, c):
    pts = [(cx, cy - 17), (cx + 15, cy - 11), (cx + 13, cy + 6), (cx, cy + 17),
           (cx - 13, cy + 6), (cx - 15, cy - 11), (cx, cy - 17)]
    cv.line(pts, c, 2.5)
    cv.line([(cx - 7, cy - 4), (cx + 7, cy - 4)], c, 2)
    cv.line([(cx - 7, cy + 3), (cx + 4, cy + 3)], c, 2)


def icon_signed_doc(cv, cx, cy, c):
    cv.rrect(cx - 12, cy - 16, 24, 32, 3, fill=PANEL, outline=c, width=2)
    cv.line([(cx - 6, cy - 8), (cx + 6, cy - 8)], c, 2)
    cv.line([(cx - 6, cy - 2), (cx + 6, cy - 2)], c, 2)
    cv.ellipse(cx + 8, cy + 11, 7, fill=c)


def icon_lock(cv, cx, cy, c):
    cv.arc(cx, cy - 4, 8, 9, 180, 360, c, 3)
    cv.line([(cx - 8, cy - 4), (cx - 8, cy + 1)], c, 3)
    cv.line([(cx + 8, cy - 4), (cx + 8, cy + 1)], c, 3)
    cv.rrect(cx - 13, cy, 26, 18, 3, fill=c)
    cv.ellipse(cx, cy + 9, 2.5, fill=PANEL)


def icon_server(cv, cx, cy, c):
    for dy in (-16, 2):
        cv.rrect(cx - 17, cy + dy, 34, 14, 3, fill=PANEL, outline=c, width=2)
        cv.ellipse(cx - 10, cy + dy + 7, 2, fill=c)
        cv.ellipse(cx - 4, cy + dy + 7, 2, fill=c)


def icon_chip(cv, cx, cy, c):
    cv.rrect(cx - 12, cy - 12, 24, 24, 3, fill=PANEL, outline=c, width=2)
    cv.rect(cx - 5, cy - 5, 10, 10, fill=c)
    for k in (-6, 0, 6):
        cv.line([(cx + k, cy - 17), (cx + k, cy - 12)], c, 2)
        cv.line([(cx + k, cy + 12), (cx + k, cy + 17)], c, 2)
        cv.line([(cx - 17, cy + k), (cx - 12, cy + k)], c, 2)
        cv.line([(cx + 12, cy + k), (cx + 17, cy + k)], c, 2)


ICONS = {1: icon_db, 2: icon_magnifier, 3: icon_network, 4: icon_shield,
         5: icon_signed_doc, 6: icon_lock, 7: icon_server, 8: icon_chip}

CAPTIONS = {
    1: ("Records enter the pipeline", "Checked against expected schema"),
    2: ("Sensitive fields detected", "and replaced with placeholders"),
    3: ("Low-rank adapter training", "on a frozen base model"),
    4: ("Pre-packaging adapter checks", "Structural + conditional behavioral"),
    5: ("Integrity metadata for the", "adapter package (not encryption)"),
    6: ("Screened package enters", "encryption module"),
    7: ("Target deployment environment", "Checks run before decryption"),
    8: ("Adapter decrypted for use", "only after checks succeed"),
}


# --------------------------------------------------------------------------- #
# Small shared widgets
# --------------------------------------------------------------------------- #
def badge(cv, cx, cy, a, color, glyph, behind=PANEL):
    cv.ellipse(cx, cy, 11, fill=fade(color, a * 0.22, behind), outline=fade(color, a, behind), width=2)
    cv.text(cx, cy + 0.5, glyph, 14, fade(color, a, behind), bold=True, anchor="mm")


def pending_ring(cv, cx, cy, a_visible, behind=PANEL):
    cv.ellipse(cx, cy, 11, outline=fade(BORDER2, a_visible, behind), width=2)


def chip(cv, x, y, w, h, label, a, outline, size=15, bold=True, tcol=TXT, fill=PANEL2, maxw=None):
    cv.rrect(x, y, w, h, 7, fill=fade(fill, a, PANEL), outline=fade(outline, a, PANEL), width=1.5)
    cv.text(x + w / 2, y + h / 2 + 0.5, label, size, fade(tcol, a, PANEL), bold=bold, anchor="mm",
            maxw=maxw or (w - 14))


def darrow(cv, x, y0, y1, a):
    col = fade((96, 120, 165), a, PANEL)
    cv.arrow([(x, y0), (x, y1)], col, width=2, head=6)


# --------------------------------------------------------------------------- #
# Stage bodies
# --------------------------------------------------------------------------- #
def body1(cv, t, F, mx, my):
    i = 1
    r_txt = R(t, F, i, 0.2)
    if r_txt > 0.001:
        cv.text(mx + 14, my + 142, "Incoming records", 14, fade(DIM, r_txt, PANEL))
    valid = P(t, i, 1.15, 0.3)
    for k in range(4):
        u = P(t, i, 0.35 + 0.15 * k, 0.3)
        a = u * F
        if a <= 0.001:
            continue
        y = my + 160 + k * 30
        dx = (1 - u) * -70
        oc = lerp(BORDER2, GREEN, valid * 0.85)
        cv.rrect(mx + 14 + dx, y, 322, 22, 6, fill=fade(PANEL2, a, PANEL), outline=fade(oc, a, PANEL), width=1.5)
        for fx, fw in ((10, 56), (82, 92), (190, 64)):
            cv.rrect(mx + 14 + dx + fx, y + 7, fw, 8, 3, fill=fade(BAR, a, PANEL))
    a = R(t, F, i, 1.0)
    if a > 0.001:
        cv.rrect(mx + 14, my + 288, 322, 32, 7, fill=fade(PANEL2, a, PANEL), outline=fade(BORDER2, a, PANEL), width=1.5)
        cv.text(mx + 28, my + 304, "Schema validation", 15, fade(TXT, a, PANEL), bold=True)
        pending_ring(cv, mx + 316, my + 304, a * (1 - valid))
        badge(cv, mx + 316, my + 304, valid * F, GREEN, "\u2713")


def body2(cv, t, F, mx, my):
    i = 2
    rows = (("email", "jane@example.com", "[EMAIL]"), ("phone", "555-0142", "[PHONE]"))
    for k, (label, raw, ph) in enumerate(rows):
        y0 = my + 140 + k * 62
        e = R(t, F, i, 0.2 + 0.1 * k)
        if e <= 0.001:
            continue
        cv.text(mx + 16, y0, label, 13, fade(DIM, e, PANEL))
        det = P(t, i, 0.35 + 0.45 * k, 0.25)
        msk = P(t, i, 0.75 + 0.45 * k, 0.3)
        ry = y0 + 12
        cv.rrect(mx + 14, ry, 322, 34, 7, fill=fade(PANEL2, e, PANEL), outline=fade(BORDER2, e, PANEL), width=1.5)
        raw_a = e * (1 - 0.65 * msk)
        w = cv.text(mx + 28, ry + 17, raw, 16, fade(TXT, raw_a, PANEL))
        hl = det * (1 - msk) * F
        if hl > 0.01:
            cv.rrect(mx + 23, ry + 5, w + 10, 24, 5, outline=fade(CYAN, hl, PANEL), width=2)
        if msk > 0.5:
            cv.line([(mx + 28, ry + 17), (mx + 28 + w, ry + 17)], fade(DIM, (msk - 0.5) * 2 * F, PANEL), 1.5)
        if msk > 0.01:
            cv.text(mx + 202, ry + 17, "\u2192", 18, fade(DIM, msk * F, PANEL), anchor="mm")
            cv.rrect(mx + 222, ry + 5, 106, 24, 6, fill=fade(PANEL, msk * F, PANEL2),
                     outline=fade(CYAN, msk * F, PANEL), width=1.5)
            cv.text(mx + 275, ry + 17.5, ph, 15, fade(CYAN, msk * F, PANEL), bold=True, anchor="mm")
    a = R(t, F, i, 1.45)
    if a > 0.001:
        cv.rrect(mx + 14, my + 278, 322, 38, 8, fill=fade(PANEL2, a, PANEL), outline=fade(CYAN, a, PANEL), width=2)
        cv.text(mx + 175, my + 297.5, "Masked Dataset", 17, fade(CYAN, a, PANEL), bold=True, anchor="mm")


def body3(cv, t, F, mx, my):
    i = 3
    a = R(t, F, i, 0.2)
    if a <= 0.001:
        return
    lora = P(t, i, 0.5, 0.6)
    dim_l = fade(BORDER2, a, PANEL)
    lit = lerp(dim_l, fade(CYAN, F, PANEL), lora)
    # base weight path (frozen)
    cv.line([(mx + 36, my + 190), (mx + 60, my + 190), (mx + 60, my + 159), (mx + 96, my + 159)], dim_l, 2)
    cv.line([(mx + 244, my + 159), (mx + 300, my + 159), (mx + 300, my + 184)], dim_l, 2)
    cv.rrect(mx + 96, my + 146, 148, 26, 6, fill=fade(PANEL2, a, PANEL), outline=dim_l, width=1.5)
    cv.text(mx + 170, my + 159.5, "W (frozen)", 13, fade(DIM, a, PANEL), anchor="mm")
    # LoRA path A -> B
    cv.line([(mx + 60, my + 190), (mx + 60, my + 209), (mx + 96, my + 209)], lit, 2)
    cv.line([(mx + 156, my + 209), (mx + 184, my + 209)], lit, 2)
    cv.line([(mx + 244, my + 209), (mx + 300, my + 209), (mx + 300, my + 196)], lit, 2)
    for bx, lab in ((96, "A"), (184, "B")):
        cv.rrect(mx + bx, my + 196, 60, 26, 6, fill=fade(PANEL2, a, PANEL), outline=lit, width=2)
        cv.text(mx + bx + 30, my + 209.5, lab, 14, lerp(fade(DIM, a, PANEL), fade(CYAN, F, PANEL), lora),
                bold=True, anchor="mm")
    pulse = 0.5 + 0.5 * math.sin(t * 6.0)
    for nx in (36, 300):
        c = lerp(fade(ICON_DIM, a, PANEL), fade(CYAN, F, PANEL), lora * (0.65 + 0.35 * pulse))
        cv.ellipse(mx + nx, my + 190, 5, fill=c)
    cv.text(mx + 170, my + 236, "trainable low-rank adapter (A, B)", 12.5,
            fade(CYAN, lora * F, PANEL), anchor="mm", maxw=300)
    # optional DP-LoRA branch (separate, dashed, clearly optional)
    d = R(t, F, i, 1.0, 0.4)
    if d > 0.001:
        pc = fade(PURPLE, d, PANEL)
        cv.dashed_line((mx + 36, my + 197), (mx + 36, my + 250), pc, 2, 5, 4)
        cv.dashed_rect(mx + 14, my + 252, 322, 68, pc, 2)
        cv.text(mx + 26, my + 270, "Optional DP-LoRA", 17, pc, bold=True)
        cv.text(mx + 26, my + 291, "Gradient clipping + Gaussian noise", 14, fade(TXT, d, PANEL), maxw=300)
        cv.text(mx + 26, my + 309, "enabled only when configured", 12, fade(DIM, d, PANEL), italic=True)


def body4(cv, t, F, mx, my):
    i = 4
    a = R(t, F, i, 0.25)
    if a <= 0.001:
        return
    cv.rrect(mx + 14, my + 140, 322, 50, 8, fill=fade(PANEL2, a, PANEL), outline=fade(CYAN, a * 0.8, PANEL), width=1.5)
    cv.text(mx + 26, my + 157, "Structural Screening", 16, fade(TXT, a, PANEL), bold=True, maxw=250)
    cv.text(mx + 26, my + 177, "Weight statistics & anomaly checks", 13, fade(DIM, a, PANEL), maxw=252)
    ok = R(t, F, i, 0.85, 0.3)
    pending_ring(cv, mx + 314, my + 165, a * (1 - P(t, i, 0.85, 0.3)))
    badge(cv, mx + 314, my + 165, ok, GREEN, "\u2713")

    b = R(t, F, i, 0.5)
    if b > 0.001:
        cv.rrect(mx + 14, my + 198, 322, 78, 8, fill=fade(PANEL2, b, PANEL))
        cv.dashed_rect(mx + 14, my + 198, 322, 78, fade(AMBER, b * 0.9, PANEL), 1.5, 7, 4)
        cv.text(mx + 26, my + 214, "Behavioral Screening", 16, fade(TXT, b, PANEL), bold=True, maxw=250)
        cv.text(mx + 26, my + 232, "Response-based testing", 13, fade(DIM, b, PANEL))
        cv.text(mx + 26, my + 250, "Requires live inference evidence", 12.5, fade(AMBER, b, PANEL), maxw=250)
        cv.text(mx + 26, my + 266, "else: Behavioral evidence unavailable", 12.5, fade(AMBER, b, PANEL),
                italic=True, maxw=290)
        badge(cv, mx + 314, my + 221, b, AMBER, "?")

    dcs = R(t, F, i, 1.1)
    if dcs > 0.001:
        cv.text(mx + 14, my + 302, "Decision", 13, fade(DIM, dcs, PANEL))
        lit = P(t, i, 1.6, 0.3) * F
        for k, (lab, col) in enumerate((("Pass", GREEN), ("Flag", AMBER), ("Reject", RED))):
            px = mx + 84 + k * 84
            base = dcs * 0.55
            f_a = lit * 0.25 if k == 0 else 0
            cv.rrect(px, my + 288, 76, 28, 8, fill=lerp(PANEL, fade(col, 1.0, PANEL), f_a),
                     outline=fade(col, base + (lit * 0.45 if k == 0 else 0), PANEL), width=1.5)
            cv.text(px + 38, my + 302.5, lab, 14, fade(col if k == 0 else TXT, base + (lit * 0.45 if k == 0 else 0), PANEL),
                    bold=(k == 0 and lit > 0.5), anchor="mm")


def body5(cv, t, F, mx, my):
    i = 5
    if R(t, F, i, 0.2) <= 0.001:
        return
    chip(cv, mx + 14, my + 140, 322, 30, "Adapter package", R(t, F, i, 0.2), CYAN, 15)
    darrow(cv, mx + 175, my + 172, my + 184, R(t, F, i, 0.45))
    d = R(t, F, i, 0.5)
    cv.rrect(mx + 14, my + 186, 322, 34, 7, fill=fade(PANEL2, d, PANEL), outline=fade(CYAN, d, PANEL), width=1.5)
    cv.text(mx + 28, my + 203, "SHA-256 Digest", 16, fade(TXT, d, PANEL), bold=True)
    for k, bw in enumerate((20, 34, 14, 28)):
        cv.rrect(mx + 190 + k * 34, my + 199, bw, 8, 3, fill=fade(BAR, d, PANEL))
    darrow(cv, mx + 175, my + 222, my + 234, R(t, F, i, 0.8))
    s = R(t, F, i, 0.9)
    cv.rrect(mx + 14, my + 236, 322, 34, 7, fill=fade(PANEL2, s, PANEL), outline=fade(CYAN, s, PANEL), width=1.5)
    cv.text(mx + 28, my + 253, "RSA-2048-PSS Signature", 16, fade(TXT, s, PANEL), bold=True, maxw=290)
    m = R(t, F, i, 1.25)
    cv.text(mx + 175, my + 289, "Integrity metadata generated", 15, fade(CYAN, m, PANEL), bold=True, anchor="mm", maxw=310)
    cv.text(mx + 175, my + 310, "authenticity & integrity, not confidentiality", 12.5, fade(DIM, m, PANEL),
            italic=True, anchor="mm", maxw=310)


def body6(cv, t, F, mx, my):
    i = 6
    if R(t, F, i, 0.2) <= 0.001:
        return
    chip(cv, mx + 14, my + 140, 322, 30, "Software-derived device fingerprint", R(t, F, i, 0.2), PURPLE, 14, maxw=300)
    darrow(cv, mx + 175, my + 172, my + 184, R(t, F, i, 0.45))
    chip(cv, mx + 14, my + 186, 322, 30, "HKDF-SHA256  \u2022  key derivation", R(t, F, i, 0.5), CYAN, 15, maxw=300)
    darrow(cv, mx + 175, my + 218, my + 230, R(t, F, i, 0.8))
    chip(cv, mx + 14, my + 232, 322, 30, "AES-256-GCM  \u2022  authenticated encryption", R(t, F, i, 0.9), CYAN, 15, maxw=300)
    darrow(cv, mx + 175, my + 264, my + 276, R(t, F, i, 1.2))
    e = R(t, F, i, 1.25)
    cv.rrect(mx + 14, my + 278, 322, 32, 7, fill=fade(PANEL2, e, PANEL), outline=fade(CYAN, e, PANEL), width=2)
    cv.text(mx + 175, my + 294.5, "Encrypted package", 16, fade(CYAN, e, PANEL), bold=True, anchor="mm")
    cv.text(mx + 175, my + 321, "Software-derived device binding", 12.5, fade(DIM, R(t, F, i, 0.3), PANEL),
            italic=True, anchor="mm")


def body7(cv, t, F, mx, my):
    i = 7
    if R(t, F, i, 0.15) <= 0.001:
        return
    labels = ("Signature valid", "Integrity digest matches", "Device binding authorized", "Replay state accepted")
    for k, lab in enumerate(labels):
        y = my + 138 + k * 37
        a = R(t, F, i, 0.15 + 0.05 * k)
        done = P(t, i, 0.35 + 0.35 * k, 0.25)
        oc = lerp(BORDER2, GREEN, done * 0.9)
        cv.rrect(mx + 14, y, 322, 31, 7, fill=fade(PANEL2, a, PANEL), outline=fade(oc, a, PANEL), width=1.5)
        cv.text(mx + 28, y + 15.5, lab, 15, fade(TXT, a, PANEL), maxw=250)
        pending_ring(cv, mx + 316, y + 15.5, a * (1 - done))
        badge(cv, mx + 316, y + 15.5, done * F, GREEN, "\u2713")
    cv.text(mx + 175, my + 305, "Any failed check \u2192 Reject Deployment", 13.5,
            fade(RED, R(t, F, i, 0.5) * 0.85, PANEL), anchor="mm", maxw=310)


def body8(cv, t, F, mx, my):
    i = 8
    if R(t, F, i, 0.2) <= 0.001:
        return
    a = R(t, F, i, 0.2)
    cv.rrect(mx + 14, my + 140, 322, 36, 8, fill=fade(PANEL2, a, PANEL), outline=fade(GREEN, a * 0.9, PANEL), width=1.5)
    cv.text(mx + 175, my + 158.5, "Authorized Adapter Loading", 16, fade(TXT, a, PANEL), bold=True, anchor="mm", maxw=300)
    darrow(cv, mx + 175, my + 178, my + 190, R(t, F, i, 0.5))
    b = R(t, F, i, 0.6)
    cv.rrect(mx + 14, my + 192, 322, 36, 8, fill=fade(PANEL2, b, PANEL), outline=fade(GREEN, b * 0.9, PANEL), width=1.5)
    cv.text(mx + 175, my + 210.5, "Inference Validation", 16, fade(TXT, b, PANEL), bold=True, anchor="mm")
    darrow(cv, mx + 175, my + 230, my + 242, R(t, F, i, 0.95))
    c = R(t, F, i, 1.0, 0.35)
    cv.rrect(mx + 14, my + 244, 322, 42, 9, fill=fade(GREEN, c * 0.22, PANEL), outline=fade(GREEN, c, PANEL), width=2)
    cv.text(mx + 175, my + 265.5, "\u2713  Deployment Accepted", 18, fade(GREEN, c, PANEL), bold=True, anchor="mm", maxw=300)
    n = R(t, F, i, 0.3)
    cv.text(mx + 175, my + 301, "Temporary plaintext files may", 12.5, fade(DIM, n, PANEL), italic=True, anchor="mm")
    cv.text(mx + 175, my + 317, "exist during loading.", 12.5, fade(DIM, n, PANEL), italic=True, anchor="mm")


BODIES = {1: body1, 2: body2, 3: body3, 4: body4, 5: body5, 6: body6, 7: body7, 8: body8}


# --------------------------------------------------------------------------- #
# Static base layer
# --------------------------------------------------------------------------- #
def build_base():
    img = Image.new("RGB", (W * S, H * S), BG)
    cv = Cv(img)
    cv.text(W / 2, 50, "SecureLoRA \u2014 Proposed End-to-End Workflow", 40, TXT, bold=True, anchor="mm")
    cv.line([(W / 2 - 90, 84), (W / 2 + 90, 84)], CYAN, 3)
    for i in range(1, 9):
        mx, my = pos(i)
        cv.rrect(mx, my, MW, MH, 12, fill=PANEL, outline=BORDER, width=2)
    return img


BASE = None

# --------------------------------------------------------------------------- #
# Arrows between modules
# --------------------------------------------------------------------------- #
def arrow_paths():
    paths = {}
    ym1, ym2 = Y1 + 165, Y2 + 165
    for k in (1, 2, 3):
        x0 = pos(k)[0] + MW + 9
        x1 = pos(k + 1)[0] - 9
        paths[k] = [(x0, ym1), (x1, ym1)]
    x_edge = XS[3] + MW + 6
    paths[4] = bezier((x_edge, ym1), (1912, ym1), (1912, ym2), (x_edge, ym2), 70)
    for k in (5, 6, 7):
        x0 = pos(k)[0] - 9
        x1 = pos(k + 1)[0] + MW + 9
        paths[k] = [(x0, ym2), (x1, ym2)]
    return paths


PATHS = arrow_paths()
ARROW_DIM = (58, 74, 110)


def draw_arrows(cv, t, F):
    for k, pts in PATHS.items():
        cv.arrow(pts, ARROW_DIM, width=4, head=14)
    for k, pts in PATHS.items():
        lit = ease((t - (T0[k + 1] - 0.55)) / 0.35) * F
        if lit <= 0:
            continue
        cv.arrow(pts, lerp(ARROW_DIM, CYAN, lit), width=4, head=14)
        for j in range(3):
            s = (t * 0.55 + j / 3.0) % 1.0
            for tail, ta in ((0.0, 1.0), (0.02, 0.55), (0.04, 0.3)):
                px, py = path_point(pts, max(0.0, s - tail))
                cv.ellipse(px, py, 4.5 - tail * 40, fill=fade((190, 235, 255), lit * ta, BG))


# --------------------------------------------------------------------------- #
# Module chrome
# --------------------------------------------------------------------------- #
def draw_module(cv, t, F, i):
    mx, my = pos(i)
    on = P(t, i, 0.0, 0.35)
    nxt = T0[i + 1] if i < 8 else T_FULL
    current = T0[i] <= t < nxt
    b = on * (1.0 if current else 0.55) * F
    glow = on * F if current else 0.0
    if glow > 0:
        for g, al in ((3, 0.35), (6, 0.16)):
            cv.rrect(mx - g, my - g, MW + 2 * g, MH + 2 * g, 12 + g, outline=fade(CYAN, glow * al, BG), width=2)
    cv.rrect(mx, my, MW, MH, 12, outline=lerp(BORDER, CYAN, b), width=2)

    # numbered circle
    cx, cy = mx + 34, my + 36
    cv.ellipse(cx, cy, 17, fill=lerp(PANEL, CYAN, b), outline=lerp(BORDER2, CYAN, b), width=2)
    cv.text(cx, cy + 0.5, str(i), 18, lerp(DIM, NAVY, b), bold=True, anchor="mm")

    # title
    tcol = lerp(TITLE_DIM, TXT, on * F)
    l1, l2 = TITLES[i]
    if l2:
        cv.text(mx + 62, my + 25, l1, 20, tcol, bold=True, maxw=276)
        cv.text(mx + 62, my + 49, l2, 20, tcol, bold=True, maxw=276)
    else:
        cv.text(mx + 62, my + 36, l1, 20, tcol, bold=True, maxw=276)
    cv.line([(mx + 14, my + 72), (mx + MW - 14, my + 72)], lerp(BORDER, BORDER2, b), 1.5)

    # icon + caption
    ic_a = P(t, i, 0.1, 0.35) * F
    ICONS[i](cv, mx + 38, my + 104, lerp(ICON_DIM, CYAN, ic_a))
    cap_a = R(t, F, i, 0.1)
    c1, c2 = CAPTIONS[i]
    cv.text(mx + 70, my + 94, c1, 14, fade(TXT, cap_a, PANEL), maxw=268)
    cv.text(mx + 70, my + 114, c2, 14, fade(DIM, cap_a, PANEL), maxw=268)

    BODIES[i](cv, t, F, mx, my)


# --------------------------------------------------------------------------- #
# Frame renderer
# --------------------------------------------------------------------------- #
def render(t):
    global BASE
    if BASE is None:
        BASE = build_base()
    F = fade_factor(t)
    img = BASE.copy()
    cv = Cv(img)

    draw_arrows(cv, t, F)
    for i in range(1, 9):
        draw_module(cv, t, F, i)

    # Reject outcome (dashed red route out of Stage 7); dim until it is relevant
    mx7, my7 = pos(7)
    r_on = ease((t - T0[7]) / 0.4) * F
    cx = mx7 + MW / 2
    rc = fade(RED, 0.20 + 0.70 * r_on, BG)
    cv.dashed_line((cx, my7 + MH + 6), (cx, my7 + MH + 28), rc, 2.5, 5, 4)
    cv.rrect(cx - 120, my7 + MH + 30, 240, 40, 10, fill=fade(RED, 0.08 * (0.3 + 0.7 * r_on), BG), outline=rc, width=2)
    cv.text(cx, my7 + MH + 50.5, "Reject Deployment", 17, rc, bold=True, anchor="mm")
    cv.text(cx + 14, my7 + MH + 18, "if any check fails", 12.5, fade(DIM, 0.25 + 0.55 * r_on, BG), italic=True, anchor="lm")

    # Legend (consistent colour semantics)
    lx, ly = 80, 903
    for col, lab, adv in ((GREEN, "Check passed", 104), (AMBER, "Conditional / unavailable", 178), (RED, "Rejected", 70)):
        cv.ellipse(lx + 6, ly, 6, fill=col)
        cv.text(lx + 20, ly, lab, 14, DIM)
        lx += adv + 34

    # Footer
    fa = ease((t - 12.6) / 0.5) * F
    cv.text(W / 2, 990, "PII Protection \u2192 Privacy-Aware Training \u2192 Adapter Screening \u2192 "
            "Cryptographic Protection \u2192 Device-Bound Deployment", 22, lerp(DIM, TXT, fa), anchor="mm")

    return img.resize((W, H), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Palette + GIF writer
# --------------------------------------------------------------------------- #
def build_palette(sample_ts):
    colors = set()
    for t in sample_ts:
        im = render(t)
        for _, c in im.getcolors(maxcolors=W * H):
            colors.add(c)
    colors = sorted(colors)
    strip = Image.new("RGB", (len(colors), 1))
    strip.putdata(colors)
    return strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def frames(pal):
    for n in range(N_FRAMES):
        im = render(n * FRAME_MS / 1000.0)
        yield im.quantize(palette=pal, dither=Image.Dither.NONE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="*", type=float, help="write PNG previews for these times (s) and exit")
    ap.add_argument("--preview-dir", default="/tmp/securelora_preview")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs/assets/securelora_workflow.gif"))
    args = ap.parse_args()

    if args.preview:
        out = Path(args.preview_dir)
        out.mkdir(parents=True, exist_ok=True)
        for t in args.preview:
            p = out / f"frame_{t:05.2f}.png"
            render(t).save(p)
            print(p)
        return

    pal = build_palette([13.2, 1.0, 3.0, 4.2, 5.9, 7.6, 8.0, 9.8, 11.2, 12.3])
    gen = frames(pal)
    first = next(gen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first.save(out, save_all=True, append_images=gen, duration=FRAME_MS, loop=0,
               optimize=False, disposal=1)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {N_FRAMES} frames, "
          f"{N_FRAMES * FRAME_MS / 1000:.2f} s)")


if __name__ == "__main__":
    main()
