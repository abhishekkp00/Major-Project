#!/usr/bin/env python3
"""
Generate a professional, high-resolution, static technical architecture diagram
for SecureLoRA ("Proposed System Architecture") on a clean light academic background.

Output: docs/assets/securelora_architecture.png (1920x1080, 16:9)
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions
W, H = 1920, 1080
S = 2  # 2x supersampling for razor-sharp typography and vector primitives

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_FONTS = {}

def font(bold, italic, size):
    key = (bold, italic, size)
    if key not in _FONTS:
        name = "DejaVuSans"
        if bold and italic:
            name += "-BoldOblique"
        elif bold:
            name += "-Bold"
        elif italic:
            name += "-Oblique"
        _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), int(round(size * S)))
    return _FONTS[key]


# Colors (Light academic palette)
BG = (248, 250, 252)          # slate-50 (pure off-white/light grey)
CARD_BG = (255, 255, 255)     # pure white
CARD_BORDER = (203, 213, 225) # slate-300
INNER_BG = (241, 245, 249)    # slate-100
INNER_BORDER = (226, 232, 240)# slate-200

NAVY_TITLE = (15, 23, 42)     # slate-900
TEXT_MAIN = (30, 41, 59)      # slate-800
TEXT_MUTED = (100, 116, 139)  # slate-500
TEXT_SUB = (71, 85, 105)      # slate-600

BLUE_ACCENT = (2, 132, 199)   # sky-600
BLUE_DARK = (30, 58, 138)     # blue-900
TEAL_ACCENT = (13, 148, 136)  # teal-600
GREEN_ACCENT = (22, 163, 74)  # emerald-600
GREEN_BG = (240, 253, 244)    # emerald-50
AMBER_ACCENT = (217, 119, 6)  # amber-600
AMBER_BG = (254, 252, 232)    # amber-50
RED_ACCENT = (220, 38, 38)    # red-600
RED_BG = (254, 242, 242)      # red-50
PURPLE_ACCENT = (126, 34, 206)# purple-700
PURPLE_BG = (250, 245, 255)   # purple-50

ARROW_COLOR = (71, 85, 105)   # slate-600


class Canvas:
    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img)

    def rrect(self, x, y, w, h, r=8, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(
            [x * S, y * S, (x + w) * S, (y + h) * S],
            radius=r * S, fill=fill, outline=outline,
            width=max(1, int(round(width * S)))
        )

    def rect(self, x, y, w, h, fill=None, outline=None, width=1):
        self.d.rectangle(
            [x * S, y * S, (x + w) * S, (y + h) * S],
            fill=fill, outline=outline, width=max(1, int(round(width * S)))
        )

    def ellipse(self, cx, cy, rx, ry=None, fill=None, outline=None, width=1):
        ry = rx if ry is None else ry
        self.d.ellipse(
            [(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
            fill=fill, outline=outline, width=max(1, int(round(width * S)))
        )

    def arc(self, cx, cy, rx, ry, a0, a1, fill, width=2):
        self.d.arc(
            [(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
            a0, a1, fill=fill, width=max(1, int(round(width * S)))
        )

    def line(self, pts, fill, width=2):
        self.d.line(
            [(x * S, y * S) for x, y in pts],
            fill=fill, width=max(1, int(round(width * S))), joint="curve"
        )

    def poly(self, pts, fill):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def text(self, x, y, s, size=15, fill=TEXT_MAIN, bold=False, italic=False, anchor="lm", maxw=None, minsize=9):
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

    def dashed_rect(self, x, y, w, h, fill, width=2, dash=7, gap=4, r=6):
        self.dashed_line((x + r, y), (x + w - r, y), fill, width, dash, gap)
        self.dashed_line((x + w, y + r), (x + w, y + h - r), fill, width, dash, gap)
        self.dashed_line((x + w - r, y + h), (x + r, y + h), fill, width, dash, gap)
        self.dashed_line((x, y + h - r), (x, y + r), fill, width, dash, gap)

    def arrow(self, pts, fill=ARROW_COLOR, width=3, head=12):
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


# Module placement
MW, MH = 370, 350
XS = [55, 520, 985, 1450]
Y1, Y2 = 120, 560

def pos(i):
    if i <= 4:
        return XS[i - 1], Y1
    return XS[8 - i], Y2


# Icons (clean flat vector style)
def draw_icon_db(cv, cx, cy, c):
    cv.ellipse(cx, cy + 10, 14, 5, fill=INNER_BG, outline=c, width=2)
    cv.rect(cx - 14, cy - 8, 28, 18, fill=INNER_BG)
    cv.line([(cx - 14, cy - 8), (cx - 14, cy + 10)], c, 2)
    cv.line([(cx + 14, cy - 8), (cx + 14, cy + 10)], c, 2)
    cv.arc(cx, cy + 1, 14, 5, 0, 180, c, 2)
    cv.arc(cx, cy - 4, 14, 5, 0, 180, c, 2)
    cv.ellipse(cx, cy - 8, 14, 5, fill=INNER_BG, outline=c, width=2)

def draw_icon_pii(cv, cx, cy, c):
    cv.rrect(cx - 14, cy - 12, 22, 26, 2, fill=INNER_BG, outline=c, width=2)
    cv.line([(cx - 9, cy - 6), (cx + 2, cy - 6)], c, 1.5)
    cv.line([(cx - 9, cy - 1), (cx + 2, cy - 1)], c, 1.5)
    cv.line([(cx - 9, cy + 4), (cx + 2, cy + 4)], c, 1.5)
    scx, scy = cx + 8, cy + 6
    pts = [(scx, scy - 9), (scx + 8, scy - 5), (scx + 7, scy + 4), (scx, scy + 10),
           (scx - 7, scy + 4), (scx - 8, scy - 5), (scx, scy - 9)]
    cv.poly(pts, fill=(255, 255, 255))
    cv.line(pts, TEAL_ACCENT, 2)
    cv.line([(scx - 3, scy + 1), (scx, scy + 4), (scx + 4, scy - 2)], TEAL_ACCENT, 2)

def draw_icon_lora(cv, cx, cy, c):
    layers = [(-14, [-8, 0, 8]), (0, [-11, -3, 5, 13]), (14, [-8, 0, 8])]
    for li in range(2):
        for ya in layers[li][1]:
            for yb in layers[li + 1][1]:
                cv.line([(cx + layers[li][0], cy + ya), (cx + layers[li + 1][0], cy + yb)], (203, 213, 225), 1)
    for lx, ys in layers:
        for ya in ys:
            cv.ellipse(cx + lx, cy + ya, 3.5, fill=c)

def draw_icon_screening(cv, cx, cy, c):
    cv.ellipse(cx - 4, cy - 4, 11, fill=INNER_BG, outline=c, width=2.5)
    cv.line([(cx + 4, cy + 4), (cx + 13, cy + 13)], c, 3.5)
    cv.ellipse(cx - 4, cy - 4, 3, fill=AMBER_ACCENT)

def draw_icon_signing(cv, cx, cy, c):
    cv.rrect(cx - 13, cy - 14, 26, 28, 2, fill=INNER_BG, outline=c, width=2)
    cv.line([(cx - 8, cy - 7), (cx + 8, cy - 7)], c, 1.5)
    cv.line([(cx - 8, cy - 2), (cx + 8, cy - 2)], c, 1.5)
    cv.ellipse(cx + 6, cy + 7, 5, fill=TEAL_ACCENT)
    cv.text(cx - 3, cy + 7, "#", 12, c, bold=True, anchor="mm")

def draw_icon_encryption(cv, cx, cy, c):
    cv.rrect(cx - 14, cy - 3, 28, 17, 2, fill=INNER_BG, outline=c, width=2)
    cv.line([(cx - 17, cy + 14), (cx + 17, cy + 14)], c, 2)
    cv.arc(cx, cy - 7, 6, 7, 180, 360, BLUE_DARK, 2)
    cv.rrect(cx - 8, cy - 4, 16, 12, 2, fill=BLUE_DARK)
    cv.ellipse(cx, cy + 2, 2, fill=(255, 255, 255))

def draw_icon_verification(cv, cx, cy, c):
    cv.rrect(cx - 13, cy - 14, 26, 28, 2, fill=INNER_BG, outline=c, width=2)
    for dy in (-7, -1, 5):
        cv.line([(cx - 8, cy + dy), (cx - 5, cy + dy + 3), (cx - 1, cy + dy - 2)], GREEN_ACCENT, 1.5)
        cv.line([(cx + 2, cy + dy + 1), (cx + 8, cy + dy + 1)], c, 1.5)

def draw_icon_loading(cv, cx, cy, c):
    cv.rrect(cx - 15, cy - 12, 30, 22, 3, fill=INNER_BG, outline=c, width=2)
    cv.line([(cx - 5, cy + 10), (cx - 8, cy + 14), (cx + 8, cy + 14), (cx + 5, cy + 10)], c, 1.5)
    cv.line([(cx - 9, cy - 3), (cx - 5, cy), (cx - 9, cy + 3)], GREEN_ACCENT, 2)
    cv.line([(cx - 2, cy + 3), (cx + 7, cy + 3)], c, 2)


ICONS = {
    1: (draw_icon_db, BLUE_DARK),
    2: (draw_icon_pii, TEAL_ACCENT),
    3: (draw_icon_lora, BLUE_DARK),
    4: (draw_icon_screening, BLUE_DARK),
    5: (draw_icon_signing, TEAL_ACCENT),
    6: (draw_icon_encryption, BLUE_DARK),
    7: (draw_icon_verification, BLUE_DARK),
    8: (draw_icon_loading, GREEN_ACCENT),
}


TITLES = {
    1: ("STAGE 1", "Dataset Ingestion & Validation"),
    2: ("STAGE 2", "PII Detection & Masking"),
    3: ("STAGE 3", "LoRA / Optional DP-LoRA Training"),
    4: ("STAGE 4", "Adapter Screening"),
    5: ("STAGE 5", "Integrity Verification & Signing"),
    6: ("STAGE 6", "Device-Bound Encryption"),
    7: ("STAGE 7", "Deployment Verification"),
    8: ("STAGE 8", "Authorized Loading & Inference"),
}


def render():
    img = Image.new("RGB", (W * S, H * S), BG)
    cv = Canvas(img)

    # ----------------------------------------------------------------------- #
    # Header
    # ----------------------------------------------------------------------- #
    cv.text(W / 2, 42, "SecureLoRA \u2014 Proposed System Architecture", 34, NAVY_TITLE, bold=True, anchor="mm")
    cv.text(W / 2, 74, "A Privacy-Preserving and Security-Screened Framework for LoRA Adapter Deployment",
            17, TEXT_SUB, italic=True, anchor="mm")
    cv.line([(W / 2 - 140, 96), (W / 2 + 140, 96)], BLUE_ACCENT, 2.5)

    # ----------------------------------------------------------------------- #
    # Inter-Module Connecting Arrows
    # ----------------------------------------------------------------------- #
    ym1 = Y1 + MH // 2
    ym2 = Y2 + MH // 2

    # Top row: 1 -> 2 -> 3 -> 4
    for k in (1, 2, 3):
        x0 = pos(k)[0] + MW + 3
        x1 = pos(k + 1)[0] - 3
        cv.arrow([(x0, ym1), (x1, ym1)], ARROW_COLOR, width=3, head=11)

    # Snake turn: 4 -> 5 (curved U-turn)
    x4_edge = pos(4)[0] + MW + 3
    x5_edge = pos(5)[0] + MW + 3
    turn_pts = bezier((x4_edge, ym1), (1890, ym1), (1890, ym2), (x5_edge, ym2), 60)
    cv.arrow(turn_pts, ARROW_COLOR, width=3, head=12)

    # Bottom row: 5 -> 6 -> 7 -> 8 (pointing left)
    for k in (5, 6, 7):
        x0 = pos(k)[0] - 3
        x1 = pos(k + 1)[0] + MW + 3
        cv.arrow([(x0, ym2), (x1, ym2)], ARROW_COLOR, width=3, head=11)

    # ----------------------------------------------------------------------- #
    # Stage Modules
    # ----------------------------------------------------------------------- #
    for i in range(1, 9):
        mx, my = pos(i)

        # Card container with subtle shadow outline
        cv.rrect(mx + 1, my + 1, MW, MH, 10, fill=CARD_BG, outline=CARD_BORDER, width=1.5)

        # Header bar
        s_lbl, s_name = TITLES[i]
        
        # Stage badge
        badge_col = BLUE_DARK if i in (1, 3, 4, 6, 7) else (TEAL_ACCENT if i in (2, 5) else GREEN_ACCENT)
        cv.rrect(mx + 14, my + 14, 76, 24, 5, fill=badge_col)
        cv.text(mx + 52, my + 26, s_lbl, 13, (255, 255, 255), bold=True, anchor="mm")

        # Icon
        draw_fn, icon_col = ICONS[i]
        draw_fn(cv, mx + MW - 28, my + 26, icon_col)

        # Title
        cv.text(mx + 14, my + 54, s_name, 17, NAVY_TITLE, bold=True, maxw=MW - 28)
        cv.line([(mx + 14, my + 70), (mx + MW - 14, my + 70)], INNER_BORDER, 1)

        # Content per stage
        if i == 1:
            # Stage 1: Dataset Ingestion & Validation
            bullets = [
                "Load source dataset",
                "Validate schema and required fields",
                "Prepare clean records",
                "Filter invalid or malformed data"
            ]
            for bi, btxt in enumerate(bullets):
                cv.ellipse(mx + 22, my + 96 + bi * 28, 3.5, fill=BLUE_ACCENT)
                cv.text(mx + 34, my + 96 + bi * 28, btxt, 14, TEXT_MAIN)

            # Output Pill
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 24, my + 300, "OUTPUT ARTIFACT", 11, TEXT_MUTED, bold=True)
            cv.text(mx + 24, my + 319, "Validated Dataset", 16, BLUE_DARK, bold=True)
            cv.ellipse(mx + MW - 36, my + 310, 8, fill=GREEN_ACCENT)
            cv.text(mx + MW - 36, my + 310, "\u2713", 11, (255, 255, 255), bold=True, anchor="mm")

        elif i == 2:
            # Stage 2: PII Detection & Masking
            bullets = [
                "Detect sensitive information",
                "Mask email, phone, and address fields",
                "Produce masked dataset securely"
            ]
            for bi, btxt in enumerate(bullets):
                cv.ellipse(mx + 22, my + 94 + bi * 25, 3.5, fill=TEAL_ACCENT)
                cv.text(mx + 34, my + 94 + bi * 25, btxt, 13.5, TEXT_MAIN)

            # Masking example box
            cv.rrect(mx + 14, my + 178, MW - 28, 88, 6, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 22, my + 192, "Sanitization Examples:", 12, TEXT_MUTED, bold=True)
            
            # Row 1
            cv.text(mx + 22, my + 215, "email: john@example.com", 12.5, TEXT_SUB)
            cv.text(mx + 215, my + 215, "\u2192", 13, TEAL_ACCENT, bold=True)
            cv.rrect(mx + 232, my + 204, 98, 22, 4, fill=(255, 255, 255), outline=TEAL_ACCENT, width=1.5)
            cv.text(mx + 281, my + 215, "[EMAIL]", 12.5, TEAL_ACCENT, bold=True, anchor="mm")

            # Row 2
            cv.text(mx + 22, my + 245, "phone: 555-0199", 12.5, TEXT_SUB)
            cv.text(mx + 215, my + 245, "\u2192", 13, TEAL_ACCENT, bold=True)
            cv.rrect(mx + 232, my + 234, 98, 22, 4, fill=(255, 255, 255), outline=TEAL_ACCENT, width=1.5)
            cv.text(mx + 281, my + 245, "[PHONE]", 12.5, TEAL_ACCENT, bold=True, anchor="mm")

            # Output Pill
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=TEAL_ACCENT, outline=TEAL_ACCENT, width=1.5)
            cv.text(mx + 24, my + 300, "OUTPUT ARTIFACT", 11, (204, 251, 241), bold=True)
            cv.text(mx + 24, my + 319, "Masked Dataset", 16, (255, 255, 255), bold=True)
            cv.text(mx + MW - 36, my + 310, "\u2713", 15, (255, 255, 255), bold=True, anchor="mm")

        elif i == 3:
            # Stage 3: LoRA / Optional DP-LoRA Training
            cv.text(mx + 14, my + 88, "Standard LoRA Training:", 13, BLUE_DARK, bold=True)
            cv.ellipse(mx + 22, my + 108, 3.5, fill=BLUE_ACCENT)
            cv.text(mx + 34, my + 108, "Freeze pretrained model weights", 13.5, TEXT_MAIN)
            cv.ellipse(mx + 22, my + 128, 3.5, fill=BLUE_ACCENT)
            cv.text(mx + 34, my + 128, "Train low-rank adapter matrices A & B", 13.5, TEXT_MAIN)

            # Optional DP-LoRA Branch (visually distinguished)
            cv.dashed_rect(mx + 14, my + 148, MW - 28, 122, PURPLE_ACCENT, width=1.5, r=6)
            cv.rrect(mx + 24, my + 140, 150, 18, 4, fill=PURPLE_BG, outline=PURPLE_ACCENT, width=1)
            cv.text(mx + 99, my + 149, "OPTIONAL BRANCH", 10.5, PURPLE_ACCENT, bold=True, anchor="mm")

            cv.text(mx + 26, my + 172, "Optional DP-LoRA Training:", 13.5, PURPLE_ACCENT, bold=True)
            cv.text(mx + 26, my + 192, "\u2022 Per-sample gradient clipping", 13, TEXT_MAIN)
            cv.text(mx + 26, my + 212, "\u2022 Calibrated Gaussian noise addition", 13, TEXT_MAIN)
            cv.text(mx + 26, my + 232, "\u2022 Differentially private privacy guarantee", 13, TEXT_MAIN)
            cv.text(mx + 26, my + 254, "Note: Optional branch; not run in every job", 11.5, TEXT_MUTED, italic=True)

            # Output Pill
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 24, my + 300, "OUTPUT ARTIFACT", 11, TEXT_MUTED, bold=True)
            cv.text(mx + 24, my + 319, "Trained LoRA Adapter", 16, BLUE_DARK, bold=True)
            cv.ellipse(mx + MW - 36, my + 310, 8, fill=GREEN_ACCENT)
            cv.text(mx + MW - 36, my + 310, "\u2713", 11, (255, 255, 255), bold=True, anchor="mm")

        elif i == 4:
            # Stage 4: Adapter Screening
            # Submodule 1: Structural Screening
            cv.rrect(mx + 14, my + 82, MW - 28, 64, 6, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 24, my + 98, "Structural Screening:", 13.5, BLUE_DARK, bold=True)
            cv.text(mx + 24, my + 116, "\u2022 Weight statistics & distribution", 12.5, TEXT_MAIN)
            cv.text(mx + 24, my + 133, "\u2022 Rank check & anomaly checks", 12.5, TEXT_MAIN)
            cv.ellipse(mx + MW - 36, my + 114, 8, fill=GREEN_ACCENT)
            cv.text(mx + MW - 36, my + 114, "\u2713", 11, (255, 255, 255), bold=True, anchor="mm")

            # Submodule 2: Behavioral Screening (conditional)
            cv.dashed_rect(mx + 14, my + 154, MW - 28, 76, AMBER_ACCENT, width=1.5, r=6)
            cv.text(mx + 24, my + 170, "Behavioral Screening (Conditional):", 13.5, AMBER_ACCENT, bold=True)
            cv.text(mx + 24, my + 188, "\u2022 Trigger testing & paraphrase consistency", 12, TEXT_MAIN)
            cv.text(mx + 24, my + 205, "\u2022 Requires real inference evidence", 12, TEXT_MAIN)
            cv.text(mx + 24, my + 221, "Unavailable evidence \u2260 successful test", 11.5, AMBER_ACCENT, italic=True)
            cv.ellipse(mx + MW - 36, my + 192, 8, fill=AMBER_ACCENT)
            cv.text(mx + MW - 36, my + 192, "?", 12, (255, 255, 255), bold=True, anchor="mm")

            # Decision gate
            cv.rrect(mx + 14, my + 240, MW - 28, 36, 6, fill=INNER_BG, outline=INNER_BORDER, width=1)
            cv.text(mx + 24, my + 258, "Decision Gate:", 12.5, NAVY_TITLE, bold=True)
            # Pass
            cv.rrect(mx + 130, my + 246, 64, 24, 4, fill=GREEN_BG, outline=GREEN_ACCENT, width=1.5)
            cv.text(mx + 162, my + 258, "Pass \u2713", 12, GREEN_ACCENT, bold=True, anchor="mm")
            # Flag
            cv.rrect(mx + 204, my + 246, 64, 24, 4, fill=AMBER_BG, outline=AMBER_ACCENT, width=1)
            cv.text(mx + 236, my + 258, "Flag", 12, AMBER_ACCENT, bold=True, anchor="mm")
            # Reject
            cv.rrect(mx + 278, my + 246, 64, 24, 4, fill=RED_BG, outline=RED_ACCENT, width=1)
            cv.text(mx + 310, my + 258, "Reject", 12, RED_ACCENT, bold=True, anchor="mm")

            # Proceed label
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=BLUE_DARK, outline=BLUE_DARK, width=1.5)
            cv.text(mx + 24, my + 300, "PASSING ADAPTERS ONLY", 10.5, (191, 219, 254), bold=True)
            cv.text(mx + 24, my + 319, "Proceed to Packaging", 15.5, (255, 255, 255), bold=True)
            cv.text(mx + MW - 36, my + 310, "\u2192", 18, (255, 255, 255), bold=True, anchor="mm")

        elif i == 5:
            # Stage 5: Integrity Verification & Signing
            cv.text(mx + 14, my + 86, "Distinct Cryptographic Operations:", 13, NAVY_TITLE, bold=True)

            # Sub-op 1: Hashing
            cv.rrect(mx + 14, my + 104, MW - 28, 56, 6, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 24, my + 120, "1. SHA-256 Package Digest", 13.5, BLUE_DARK, bold=True)
            cv.text(mx + 24, my + 140, "Fixed 256-bit cryptographic digest", 13, TEXT_MAIN)
            cv.ellipse(mx + MW - 36, my + 132, 8, fill=BLUE_ACCENT)
            cv.text(mx + MW - 36, my + 132, "#", 11, (255, 255, 255), bold=True, anchor="mm")

            # Sub-op 2: Digital Signature
            cv.rrect(mx + 14, my + 168, MW - 28, 64, 6, fill=INNER_BG, outline=INNER_BORDER, width=1.5)
            cv.text(mx + 24, my + 184, "2. RSA-2048-PSS Signature", 13.5, TEAL_ACCENT, bold=True)
            cv.text(mx + 24, my + 204, "Digitally signed with private key", 13, TEXT_MAIN)
            cv.text(mx + 24, my + 221, "Authenticity & Non-repudiation", 11.5, TEXT_MUTED, italic=True)
            cv.ellipse(mx + MW - 36, my + 200, 8, fill=TEAL_ACCENT)
            cv.text(mx + MW - 36, my + 200, "\u270d", 11, (255, 255, 255), bold=True, anchor="mm")

            cv.text(mx + 14, my + 252, "\u2022 Generates integrity metadata package", 12.5, TEXT_SUB)
            cv.text(mx + 14, my + 270, "\u2022 Signing provides authenticity, not encryption", 11.5, TEXT_MUTED, italic=True)

            # Output Pill
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=TEAL_ACCENT, outline=TEAL_ACCENT, width=1.5)
            cv.text(mx + 24, my + 300, "OUTPUT ARTIFACT", 11, (204, 251, 241), bold=True)
            cv.text(mx + 24, my + 319, "Signed Adapter Package", 15.5, (255, 255, 255), bold=True)
            cv.text(mx + MW - 36, my + 310, "\u2713", 15, (255, 255, 255), bold=True, anchor="mm")

        elif i == 6:
            # Stage 6: Device-Bound Encryption
            bullets = [
                ("Device Fingerprint:", "Software-derived device identity"),
                ("Key Derivation:", "HKDF-SHA256 expansion"),
                ("Authenticated Cipher:", "AES-256-GCM (AEAD mode)"),
            ]
            for bi, (bt1, bt2) in enumerate(bullets):
                y = my + 88 + bi * 44
                cv.rrect(mx + 14, y, MW - 28, 38, 5, fill=INNER_BG, outline=INNER_BORDER, width=1.2)
                cv.text(mx + 24, y + 14, bt1, 12, TEXT_MUTED, bold=True)
                cv.text(mx + 24, y + 28, bt2, 13.5, NAVY_TITLE, bold=True)

            # Explicit qualification note
            cv.rrect(mx + 14, my + 228, MW - 28, 48, 5, fill=AMBER_BG, outline=AMBER_ACCENT, width=1)
            cv.text(mx + 24, my + 243, "\u26a0 Security Clarification:", 12, AMBER_ACCENT, bold=True)
            cv.text(mx + 24, my + 261, "Software-derived binding; not hardware-backed", 12, TEXT_MAIN)

            # Output Pill
            cv.rrect(mx + 14, my + 288, MW - 28, 44, 7, fill=BLUE_DARK, outline=BLUE_DARK, width=1.5)
            cv.text(mx + 24, my + 300, "OUTPUT ARTIFACT", 11, (191, 219, 254), bold=True)
            cv.text(mx + 24, my + 319, "Encrypted Adapter Package", 15, (255, 255, 255), bold=True)
            cv.text(mx + MW - 36, my + 310, "\U0001f512", 14, (255, 255, 255), bold=True, anchor="mm")

        elif i == 7:
            # Stage 7: Deployment Verification
            cv.text(mx + 14, my + 86, "Sequential Verification Gates:", 13, NAVY_TITLE, bold=True)

            checks = [
                "Verify package signature",
                "Verify integrity digest",
                "Check device authorization",
                "Check replay state"
            ]
            for ci, ctxt in enumerate(checks):
                y = my + 106 + ci * 32
                cv.rrect(mx + 14, y, MW - 28, 27, 4, fill=INNER_BG, outline=INNER_BORDER, width=1)
                cv.text(mx + 24, y + 13.5, f"[ {ci+1} ]  {ctxt}", 13, TEXT_MAIN)
                cv.ellipse(mx + MW - 32, y + 13.5, 7, fill=GREEN_ACCENT)
                cv.text(mx + MW - 32, y + 13.5, "\u2713", 10, (255, 255, 255), bold=True, anchor="mm")

            cv.text(mx + 14, my + 248, "All checks run before decryption occurs.", 11.5, TEXT_MUTED, italic=True)

            # Two outcomes:
            cv.rrect(mx + 14, my + 266, (MW - 34) // 2, 66, 6, fill=GREEN_BG, outline=GREEN_ACCENT, width=1.5)
            cv.text(mx + 24, my + 282, "All Checks Pass", 12.5, GREEN_ACCENT, bold=True)
            cv.text(mx + 24, my + 304, "Continue \u2192 Stage 8", 14, GREEN_ACCENT, bold=True)

            cv.rrect(mx + 14 + (MW - 34) // 2 + 6, my + 266, (MW - 34) // 2, 66, 6, fill=RED_BG, outline=RED_ACCENT, width=1.5)
            cv.text(mx + 24 + (MW - 34) // 2 + 6, my + 282, "Any Check Fails", 12.5, RED_ACCENT, bold=True)
            cv.text(mx + 24 + (MW - 34) // 2 + 6, my + 304, "Reject Deployment", 13.5, RED_ACCENT, bold=True)

        elif i == 8:
            # Stage 8: Authorized Loading & Inference Validation
            steps = [
                ("1. Authorize adapter loading", "Decrypt payload into RAM/tmp fs"),
                ("2. Load adapter for inference", "Attach LoRA weights to base model"),
                ("3. Validate inference behavior", "Smoke test & behavioral validation")
            ]
            for si, (st1, st2) in enumerate(steps):
                y = my + 86 + si * 46
                cv.rrect(mx + 14, y, MW - 28, 40, 5, fill=INNER_BG, outline=INNER_BORDER, width=1.2)
                cv.text(mx + 24, y + 14, st1, 13, BLUE_DARK, bold=True)
                cv.text(mx + 24, y + 29, st2, 12, TEXT_MUTED)

            # Caution note
            cv.text(mx + 14, my + 240, "\u2022 Temporary plaintext files may exist", 11.5, TEXT_MUTED, italic=True)
            cv.text(mx + 14, my + 256, "  during adapter loading.", 11.5, TEXT_MUTED, italic=True)

            # Final Success Outcome
            cv.rrect(mx + 14, my + 282, MW - 28, 52, 8, fill=GREEN_ACCENT, outline=GREEN_ACCENT, width=2)
            cv.text(mx + 24, my + 298, "FINAL SUCCESS OUTCOME", 11, (220, 252, 231), bold=True)
            cv.text(mx + 24, my + 320, "Deployment Accepted", 18, (255, 255, 255), bold=True)
            cv.ellipse(mx + MW - 36, my + 308, 12, fill=(255, 255, 255))
            cv.text(mx + MW - 36, my + 308, "\u2713", 16, GREEN_ACCENT, bold=True, anchor="mm")

    # ----------------------------------------------------------------------- #
    # Rejection Outlets (Explicit rejection arrows)
    # ----------------------------------------------------------------------- #
    # Rejection from Stage 4 (Screening failed)
    mx4, my4 = pos(4)
    rx4 = mx4 + MW // 2
    cv.dashed_line((rx4, my4 + MH), (rx4, my4 + MH + 26), RED_ACCENT, 2, 5, 4)
    cv.rrect(rx4 - 90, my4 + MH + 26, 180, 32, 6, fill=RED_BG, outline=RED_ACCENT, width=1.5)
    cv.text(rx4, my4 + MH + 42, "\u2717  Screening Rejected", 13.5, RED_ACCENT, bold=True, anchor="mm")

    # Rejection from Stage 7 (Deployment failed)
    mx7, my7 = pos(7)
    rx7 = mx7 + (MW - 34) // 2 + 6 + (MW - 34) // 4
    cv.dashed_line((rx7, my7 + MH), (rx7, my7 + MH + 26), RED_ACCENT, 2, 5, 4)
    cv.rrect(rx7 - 85, my7 + MH + 26, 170, 32, 6, fill=RED_BG, outline=RED_ACCENT, width=1.5)
    cv.text(rx7, my7 + MH + 42, "\u2717  Reject Deployment", 13.5, RED_ACCENT, bold=True, anchor="mm")

    # ----------------------------------------------------------------------- #
    # Bottom Summary Ribbon
    # ----------------------------------------------------------------------- #
    ribbon_y = 985
    cv.rrect(120, ribbon_y, W - 240, 56, 10, fill=(241, 245, 249), outline=CARD_BORDER, width=1.5)

    stages_summary = [
        ("PII Protection", TEAL_ACCENT),
        ("Privacy-Aware Training", PURPLE_ACCENT),
        ("Adapter Screening", BLUE_DARK),
        ("Cryptographic Protection", TEAL_ACCENT),
        ("Device-Bound Deployment", GREEN_ACCENT)
    ]

    sx = 175
    for idx, (s_text, s_col) in enumerate(stages_summary):
        cv.rrect(sx, ribbon_y + 12, 220, 32, 6, fill=(255, 255, 255), outline=s_col, width=1.5)
        cv.text(sx + 110, ribbon_y + 28, s_text, 13, s_col, bold=True, anchor="mm")
        if idx < len(stages_summary) - 1:
            cv.text(sx + 248, ribbon_y + 28, "\u2794", 17, TEXT_MUTED, anchor="mm")
        sx += 300

    # Downsample using LANCZOS to 1920x1080
    return img.resize((W, H), Image.LANCZOS)


def main():
    out_path = Path("/home/abhishek/Projects/MAJOR_PROJECT/docs/assets/securelora_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render()
    img.save(out_path, format="PNG")
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
