# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""Generate a filmstrip PNG showing the 3 Council Deliberation stages as panels."""
from PIL import Image, ImageDraw, ImageFont
import os

# Panel dimensions
PW, PH = 760, 420
GAP = 30          # vertical gap between panels
PAD = 30          # outer padding
ARROW_H = 36      # height reserved for arrow between panels

# Total canvas
W = PW + PAD * 2
H = PAD + PH * 3 + ARROW_H * 2 + GAP * 2 + PAD + 60  # +60 for top title

BG = "#ffffff"
PRIMARY = "#111111"
SECONDARY = "#555555"
MUTED = "#999999"
BORDER = "#e5e5e5"
ACCENT_BG = "#f5f5f5"
GREEN = "#22863a"
BLUE = "#0366d6"
RED = "#d73a49"

# Fonts
try:
    FONT_BOLD = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    FONT_REG = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    FONT_STAGE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    FONT_ARROW = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
except Exception:
    FONT_BOLD = FONT_REG = FONT_SMALL = FONT_TITLE = FONT_STAGE = FONT_ARROW = ImageFont.load_default()


def rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


EXPERTS = [
    ("Statistician", "Strong upward trend (R\u00b2=0.72). Seasonal cycle of 12 months detected. Forecast: +8.2% growth."),
    ("Domain Expert", "Pattern consistent with Q4 retail seasonality. Holiday effect expected in Dec. Forecast: +9.1%."),
    ("Risk Analyst", "Volatility increasing. 15% chance of demand shock. Downside risk: -3.4%. Recommend buffer."),
    ("Opportunity Analyst", "Market expansion signal. New customer segment growing 22% YoY. Upside: +12.5%."),
    ("Chief Analyst", "Reviewing all perspectives before synthesis. Noting divergence on risk assessment."),
]

PEER_SCORES = [
    ("Statistician", ["-", "4", "3", "4", "5"]),
    ("Domain Expert", ["5", "-", "3", "4", "4"]),
    ("Risk Analyst", ["4", "3", "-", "2", "4"]),
    ("Opportunity", ["3", "4", "3", "-", "4"]),
    ("Chief Analyst", ["5", "4", "4", "4", "-"]),
]

SYNTHESIS = (
    "CONSENSUS: Growth of +8 to 9% expected (4/5 experts agree).\n"
    "KEY RISK: Demand shock (15% probability, flagged by Risk Analyst).\n"
    "RECOMMENDATION: Plan for +8.5% growth with 5% downside buffer.\n"
    "CONFIDENCE: High (Borda score: Statistician 17, Domain 15, Risk 13)"
)


def draw_down_arrow(draw, cx, top_y, label):
    """Draw a downward arrow with a label between panels."""
    arrow_len = ARROW_H - 8
    # Shaft
    draw.line([(cx, top_y + 4), (cx, top_y + arrow_len - 6)], fill=MUTED, width=2)
    # Arrowhead
    draw.polygon([
        (cx, top_y + arrow_len),
        (cx - 6, top_y + arrow_len - 10),
        (cx + 6, top_y + arrow_len - 10),
    ], fill=MUTED)
    # Label
    draw.text((cx + 14, top_y + 6), label, fill=MUTED, font=FONT_ARROW)


# ── Build the filmstrip ──────────────────────────────────────────────
img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# Outer border
rounded_rect(draw, [0, 0, W - 1, H - 1], radius=16, outline=BORDER, width=2)

# Title
draw.text((W // 2, PAD + 10), "Council Deliberation \u2014 Sample Output", fill=PRIMARY, font=FONT_TITLE, anchor="mm")
title_bottom = PAD + 30

# ── PANEL 1: Stage 1 — First Opinions (all 5 experts) ────────────
p1_y = title_bottom + 16
rounded_rect(draw, [PAD, p1_y, PAD + PW, p1_y + PH], radius=12, fill=BG, outline=BORDER, width=2)

# Stage label with colored badge
badge_x, badge_y = PAD + 16, p1_y + 12
rounded_rect(draw, [badge_x, badge_y, badge_x + 80, badge_y + 24], radius=6, fill=PRIMARY)
draw.text((badge_x + 40, badge_y + 12), "Stage 1", fill="#ffffff", font=FONT_BOLD, anchor="mm")
draw.text((badge_x + 92, badge_y + 4), "First Opinions", fill=PRIMARY, font=FONT_STAGE)

# Expert cards
colors = ["#0366d6", "#22863a", "#d73a49", "#e36209", "#6f42c1"]
card_y = p1_y + 48
for i, (name, opinion) in enumerate(EXPERTS):
    cy = card_y + i * 68
    rounded_rect(draw, [PAD + 16, cy, PAD + PW - 16, cy + 58], radius=8, fill=ACCENT_BG, outline=BORDER)
    # Colored dot
    draw.ellipse([PAD + 26, cy + 8, PAD + 40, cy + 22], fill=colors[i])
    draw.text((PAD + 48, cy + 6), name, fill=PRIMARY, font=FONT_BOLD)
    # Opinion text (truncated)
    draw.text((PAD + 48, cy + 28), opinion[:90] + ("..." if len(opinion) > 90 else ""), fill=SECONDARY, font=FONT_SMALL)

# Status
draw.text((PAD + PW - 120, p1_y + PH - 24), "5/5 experts complete", fill=GREEN, font=FONT_SMALL)

# ── Arrow 1 → 2 ──────────────────────────────────────────────────
arrow1_y = p1_y + PH + 2
draw_down_arrow(draw, W // 2, arrow1_y, "Experts rank each other's analyses")

# ── PANEL 2: Stage 2 — Peer Review ───────────────────────────────
p2_y = arrow1_y + ARROW_H + 2
rounded_rect(draw, [PAD, p2_y, PAD + PW, p2_y + PH], radius=12, fill=BG, outline=BORDER, width=2)

# Stage label
badge_y2 = p2_y + 12
rounded_rect(draw, [PAD + 16, badge_y2, PAD + 96, badge_y2 + 24], radius=6, fill=PRIMARY)
draw.text((PAD + 56, badge_y2 + 12), "Stage 2", fill="#ffffff", font=FONT_BOLD, anchor="mm")
draw.text((PAD + 108, badge_y2 + 4), "Peer Review", fill=PRIMARY, font=FONT_STAGE)

# Score matrix
headers = ["", "Stat", "Domain", "Risk", "Opp", "Chief"]
col_w = 100
row_h = 28
table_x = PAD + 55
table_y = p2_y + 52

# Header row
for j, h in enumerate(headers):
    x = table_x + j * col_w
    draw.rectangle([x, table_y, x + col_w, table_y + row_h], fill="#f0f0f0", outline=BORDER)
    draw.text((x + col_w // 2, table_y + row_h // 2), h, fill=PRIMARY, font=FONT_BOLD, anchor="mm")

# Score rows
for i, (name, scores) in enumerate(PEER_SCORES):
    ry = table_y + (i + 1) * row_h
    draw.rectangle([table_x, ry, table_x + col_w, ry + row_h], fill="#f8f8f8", outline=BORDER)
    draw.text((table_x + col_w // 2, ry + row_h // 2), name, fill=PRIMARY, font=FONT_SMALL, anchor="mm")
    for j, s in enumerate(scores):
        x = table_x + (j + 1) * col_w
        bg_c = "#e8f5e9" if s in ("4", "5") else ("#fff3e0" if s in ("2", "3") else ACCENT_BG)
        draw.rectangle([x, ry, x + col_w, ry + row_h], fill=bg_c, outline=BORDER)
        draw.text((x + col_w // 2, ry + row_h // 2), s, fill=PRIMARY if s != "-" else MUTED, font=FONT_REG, anchor="mm")

# Borda scores
borda_y = table_y + 6 * row_h + 16
draw.text((table_x, borda_y), "Borda Count:", fill=PRIMARY, font=FONT_BOLD)
borda_scores = "Statistician: 17  |  Domain Expert: 15  |  Risk: 13  |  Opportunity: 14  |  Chief: 17"
draw.text((table_x + 120, borda_y), borda_scores, fill=SECONDARY, font=FONT_SMALL)

# Ranking
rank_y = borda_y + 28
draw.text((table_x, rank_y), "Ranking:", fill=PRIMARY, font=FONT_BOLD)
draw.text((table_x + 80, rank_y), "1. Statistician / Chief (17)    2. Domain Expert (15)    3. Opportunity (14)    4. Risk (13)", fill=SECONDARY, font=FONT_SMALL)

# Status
draw.text((PAD + PW - 120, p2_y + PH - 24), "Peer review complete", fill=GREEN, font=FONT_SMALL)

# ── Arrow 2 → 3 ──────────────────────────────────────────────────
arrow2_y = p2_y + PH + 2
draw_down_arrow(draw, W // 2, arrow2_y, "Chairman synthesizes all reviews")

# ── PANEL 3: Stage 3 — Chairman Synthesis ─────────────────────────
p3_y = arrow2_y + ARROW_H + 2
rounded_rect(draw, [PAD, p3_y, PAD + PW, p3_y + PH], radius=12, fill=BG, outline=BORDER, width=2)

# Stage label
badge_y3 = p3_y + 12
rounded_rect(draw, [PAD + 16, badge_y3, PAD + 96, badge_y3 + 24], radius=6, fill=PRIMARY)
draw.text((PAD + 56, badge_y3 + 12), "Stage 3", fill="#ffffff", font=FONT_BOLD, anchor="mm")
draw.text((PAD + 108, badge_y3 + 4), "Chairman Synthesis", fill=PRIMARY, font=FONT_STAGE)

# Stage 1 & 2 completed labels
draw.text((PAD + PW - 200, badge_y3 + 4), "Stage 1: Complete", fill=GREEN, font=FONT_SMALL)
draw.text((PAD + PW - 200, badge_y3 + 18), "Stage 2: Complete", fill=GREEN, font=FONT_SMALL)

# Synthesis box
syn_y = p3_y + 52
rounded_rect(draw, [PAD + 16, syn_y, PAD + PW - 16, syn_y + 170], radius=10, fill="#f0f7ff", outline="#c8ddf0", width=2)
draw.text((PAD + 30, syn_y + 10), "Final Recommendation \u2014 Chief Analyst", fill=PRIMARY, font=FONT_BOLD)

# Synthesis text lines
lines = SYNTHESIS.split("\n")
for i, line in enumerate(lines):
    ly = syn_y + 40 + i * 26
    if line.startswith("CONSENSUS"):
        draw.text((PAD + 30, ly), line, fill=GREEN, font=FONT_REG)
    elif line.startswith("KEY RISK"):
        draw.text((PAD + 30, ly), line, fill=RED, font=FONT_REG)
    elif line.startswith("RECOMMENDATION"):
        draw.text((PAD + 30, ly), line, fill=BLUE, font=FONT_REG)
    else:
        draw.text((PAD + 30, ly), line, fill=SECONDARY, font=FONT_REG)

# Confidence bar
conf_y = syn_y + 145
draw.text((PAD + 30, conf_y), "Confidence: HIGH", fill=PRIMARY, font=FONT_BOLD)
bar_x = PAD + 180
for seg in range(10):
    color = GREEN if seg < 8 else BORDER
    draw.rectangle([bar_x + seg * 38, conf_y + 2, bar_x + seg * 38 + 34, conf_y + 14], fill=color)

# Deliberation complete status
draw.text((PAD + 16, p3_y + PH - 36), "Deliberation complete \u2014 3/3 stages finished", fill=GREEN, font=FONT_BOLD)
# Full green bar
rounded_rect(draw, [PAD + 16, p3_y + PH - 16, PAD + PW - 16, p3_y + PH - 8], radius=4, fill=GREEN)

# ── Save ──────────────────────────────────────────────────────────
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, "council_deliberation_demo.png")
img.save(output_path, "PNG", optimize=True)
print(f"Filmstrip saved to {output_path}")
print(f"Size: {os.path.getsize(output_path)} bytes")
print(f"Dimensions: {W}x{H}")
