"""AI photo culling: group similar JPEGs, then let an LLM pick the best of each group."""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

BATCH = 6  # max photos per match
ADVANCE = 2  # photos advancing from each match
TOP = 3  # candidates inspected with 100% crops
CROP = 1024  # crop size in original pixels
TILE = 512  # grid tile size
ISO_MAX = 3200
FLAGS = ["closed_eyes", "cropped_limbs", "bad_expression"]
PROMPT_VERSION = 1
HERE = Path(__file__).resolve().parent
TASTE = HERE / "taste.md"
OVERRIDES = HERE / "overrides.jsonl"


# ---------- step 0: read + step 2: technical scores ----------

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def open_image(path, size):
    im = Image.open(path)
    scale = max(im.size) / size
    im.draft("RGB", (int(im.width / scale), int(im.height / scale)))  # fast JPEG decode at reduced scale
    return ImageOps.exif_transpose(im).convert("RGB")


def read_exif(im, path):
    exif = im.getexif()
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    try:
        taken = sub[ExifTags.Base.DateTimeOriginal].strip("\x00 ")
        ts = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S").timestamp()
    except (KeyError, ValueError):  # missing or unset camera clock, e.g. 0000:00:00
        ts = path.stat().st_mtime
    else:
        subsec = str(sub.get(ExifTags.Base.SubsecTimeOriginal, "")).strip()
        if subsec.isdigit():
            ts += float("0." + subsec)
    iso = sub.get(ExifTags.Base.ISOSpeedRatings)
    if isinstance(iso, tuple):
        iso = iso[0]
    num = lambda v: float(v) if v else None
    return {
        "time": ts,
        "shutter": num(sub.get(ExifTags.Base.ExposureTime)),
        "focal": num(sub.get(ExifTags.Base.FocalLengthIn35mmFilm) or sub.get(ExifTags.Base.FocalLength)),
        "iso": int(iso) if iso else None,
    }


def dhash(im):
    px = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = bits << 1 | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def sharpness(gray):
    lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    w, h = lap.size
    return max(
        ImageStat.Stat(lap.crop((c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3))).var[0]
        for r in range(3) for c in range(3)
    )


def analyze(path, work):
    sha = file_hash(path)
    cache = work / "photos" / f"{sha}.json"
    if cache.exists():
        return {**json.loads(cache.read_text()), "file": path.name}
    with Image.open(path) as raw:
        exif = read_exif(raw, path)
    im = open_image(path, 2048)
    gray = im.convert("L")
    hist = gray.histogram()
    total = sum(hist)
    risks = []
    if exif["shutter"] and exif["focal"] and exif["shutter"] > 1 / exif["focal"]:
        risks.append("slow_shutter")
    if exif["iso"] and exif["iso"] > ISO_MAX:
        risks.append("high_iso")
    info = {
        "sha": sha,
        **exif,
        "dhash": dhash(im),
        "sharpness": sharpness(gray),
        "over": sum(hist[250:]) / total,
        "under": sum(hist[:6]) / total,
        "risks": risks,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(info))
    return {**info, "file": path.name}


def scan(folder, work, jobs):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    work.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(jobs) as pool:
        return {p["file"]: p for p in pool.map(lambda p: analyze(p, work), paths)}


# ---------- step 1: grouping ----------

def make_groups(photos, gap, max_dist):
    groups = []
    prev = None
    for p in sorted(photos.values(), key=lambda p: (p["time"], p["file"])):
        if prev and p["time"] - prev["time"] <= gap and (p["dhash"] ^ prev["dhash"]).bit_count() <= max_dist:
            groups[-1].append(p["file"])
        else:
            groups.append([p["file"]])
        prev = p
    return groups


# ---------- LLM ----------

PI_SLOTS = threading.Semaphore(4)  # concurrent pi calls; cmd_rank sets it from --jobs


def ask_pi(model, images, prompt):
    cmd = ["pi", "-p", "--no-session", "--no-tools", "-nc", "-ns", "--model", model,
           *[f"@{p}" for p in images], prompt]
    with PI_SLOTS:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def ask_json(model, images, prompt, labels):
    """Ask pi for JSON with a "ranking" that is a permutation of labels; retry once."""
    for attempt in range(2):
        try:
            out = ask_pi(model, images, prompt)
            data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            data["ranking"] = [int(x) for x in data["ranking"]]
            if sorted(data["ranking"]) != sorted(labels):
                raise ValueError(f"bad ranking {data['ranking']} for labels {labels}")
            return data
        except (subprocess.CalledProcessError, AttributeError, ValueError, KeyError, TypeError) as e:
            err = getattr(e, "stderr", None) or e
            print(f"pi attempt {attempt + 1} failed: {err}", file=sys.stderr)
    raise RuntimeError("pi did not return valid JSON")


def label_image(im, label):
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(24, im.width // 10))
    box = draw.textbbox((0, 0), str(label), font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill="black")
    draw.text((8, 8), str(label), fill="white", font=font)
    return im


def grid(paths, out):
    cols = min(3, len(paths))
    rows = -(-len(paths) // cols)
    sheet = Image.new("RGB", (cols * TILE, rows * TILE), "gray")
    for i, path in enumerate(paths):
        im = open_image(path, TILE)
        im.thumbnail((TILE, TILE))
        tile = Image.new("RGB", (TILE, TILE), "gray")
        tile.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
        sheet.paste(label_image(tile, i + 1), (i % cols * TILE, i // cols * TILE))
    sheet.save(out, quality=90)


def crop(path, cell, label, out):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    row, col = divmod(min(max(int(cell), 1), 9) - 1, 3)
    cx, cy = (col + 0.5) * im.width / 3, (row + 0.5) * im.height / 3
    x = int(min(max(cx - CROP / 2, 0), max(im.width - CROP, 0)))
    y = int(min(max(cy - CROP / 2, 0), max(im.height - CROP, 0)))
    label_image(im.crop((x, y, x + CROP, y + CROP)), label).save(out, quality=92)


ROUND1 = """You are culling photos. The attached image is a grid of {n} photos of the same scene, each labelled with a number in its top-left corner.
Rank them from best to worst. Judge focus on the subject, exposure, composition, the moment, expressions, closed eyes, and cropped hands or feet.

Technical scores, relative within this set (sharpness 100 = sharpest here; over/under = % of clipped pixels):
{scores}

User taste notes:
{taste}

Divide each photo into a 3x3 grid of cells numbered 1-9 row by row from the top-left. For each of your top {top} photos, give the cell with the most important detail to inspect at 100% (for example the main face).

Reply with JSON only, no prose and no code fence:
{{"ranking": [labels, best first], "flags": {{"label": [any of {flags}]}}, "focus": {{"label": cell}}}}"""

ROUND2 = """You are culling photos. The attached images are 100% crops of the key region of photos {labels}, each labelled with its number in the top-left corner.
Your earlier overall ranking of these photos was {labels}. Re-rank them, now also judging fine detail: focus on the subject, open eyes, expression, motion blur, noise.

Reply with JSON only, no prose and no code fence:
{{"ranking": [labels, best first], "flags": {{"label": [any of {flags}]}}}}"""


def score_lines(photos):
    top = max(p["sharpness"] for p in photos) or 1
    return "\n".join(
        f"{i + 1}: sharpness {round(100 * p['sharpness'] / top)}, over {p['over']:.1%}, under {p['under']:.1%}"
        + (f", risks: {', '.join(p['risks'])}" if p["risks"] else "")
        for i, p in enumerate(photos)
    )


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def merge_flags(flags, label_to_file, into):
    """Collect LLM flags per file; malformed entries are ignored."""
    if not isinstance(flags, dict):
        return
    for label, items in flags.items():
        file = label_to_file.get(to_int(label))
        if file and isinstance(items, list):
            into.setdefault(file, set()).update(f for f in items if f in FLAGS)


def match(ctx, photos, run):
    """One match: shuffled grid -> ranking + focus cells -> 100% crops of top -> re-rank."""
    key = hashlib.sha256(json.dumps(
        [sorted(p["sha"] for p in photos), run, ctx["model"], ctx["taste_hash"], PROMPT_VERSION]
    ).encode()).hexdigest()[:16]
    cache = ctx["work"] / "matches" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    order = sorted(photos, key=lambda p: p["sha"])
    random.Random(key).shuffle(order)  # position bias: new order every match
    labels = list(range(1, len(order) + 1))
    by_label = {i + 1: p["file"] for i, p in enumerate(order)}
    img = ctx["work"] / "img"
    img.mkdir(parents=True, exist_ok=True)

    sheet = img / f"{key}-grid.jpg"
    grid([ctx["folder"] / p["file"] for p in order], sheet)
    r1 = ask_json(ctx["model"], [sheet], ROUND1.format(
        n=len(order), scores=score_lines(order), taste=ctx["taste"] or "(none)",
        top=min(TOP, len(order)), flags=FLAGS), labels)

    top = r1["ranking"][:TOP]
    focus = r1.get("focus")
    focus = {to_int(k): to_int(v) for k, v in focus.items()} if isinstance(focus, dict) else {}
    crops = []
    for label in top:
        out = img / f"{key}-crop{label}.jpg"
        crop(ctx["folder"] / by_label[label], focus.get(label) or 5, label, out)
        crops.append(out)
    r2 = ask_json(ctx["model"], crops, ROUND2.format(labels=top, flags=FLAGS), top)

    flags = {}
    merge_flags(r1.get("flags"), by_label, flags)
    merge_flags(r2.get("flags"), by_label, flags)
    result = {
        "ranking": [by_label[l] for l in r2["ranking"] + r1["ranking"][TOP:]],
        "flags": {f: sorted(v) for f, v in flags.items()},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


# ---------- step 3: tournament ----------

def split_even(items):
    n = -(-len(items) // BATCH)
    size, extra = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + size + (i < extra)
        out.append(items[start:end])
        start = end
    return out


def tournament(ctx, group):
    photos = [ctx["photos"][f] for f in group]
    if len(photos) == 1:
        return {"best": group[0], "ranking": group, "flags": {}, "needs_review": False}
    eliminated = []  # per round, losers ordered by their in-match rank
    flags = {}
    while len(photos) > BATCH:
        batches = split_even(photos)
        with ThreadPoolExecutor(len(batches)) as pool:
            results = list(pool.map(lambda b: match(ctx, b, 0), batches))
        for r in results:
            for f, v in r["flags"].items():
                flags.setdefault(f, set()).update(v)
        losers = sorted((i, f) for r in results for i, f in enumerate(r["ranking"]) if i >= ADVANCE)
        eliminated.append([f for _, f in losers])
        photos = [ctx["photos"][f] for r in results for f in r["ranking"][:ADVANCE]]

    # final runs twice with different shuffles; disagreement -> user review
    with ThreadPoolExecutor(2) as pool:
        first, second = pool.map(lambda run: match(ctx, photos, run), (0, 1))
    for r in (first, second):
        for f, v in r["flags"].items():
            flags.setdefault(f, set()).update(v)
    ranking = first["ranking"] + [f for losers in reversed(eliminated) for f in losers]
    return {
        "best": ranking[0],
        "ranking": ranking,
        "flags": {f: sorted(v) for f, v in flags.items()},
        "needs_review": first["ranking"][0] != second["ranking"][0],
        "final_alt_best": second["ranking"][0],
    }


# ---------- commands ----------

def workdir(folder):
    folder = folder.resolve()
    return HERE / "runs" / f"{folder.name}-{hashlib.sha256(str(folder).encode()).hexdigest()[:8]}"


def cmd_group(args):
    work = workdir(args.folder)
    photos = scan(args.folder, work, args.jobs)
    groups = make_groups(photos, args.gap, args.dhash)
    (work / "groups.json").write_text(json.dumps(groups, indent=1, ensure_ascii=False))
    print(f"{len(photos)} photos -> {len(groups)} groups: {work / 'groups.json'}")
    print("Edit that file to split or merge groups, then run: uv run pic_pick.py rank", args.folder)
    return photos


def cmd_rank(args):
    global PI_SLOTS
    PI_SLOTS = threading.Semaphore(args.jobs)
    work = workdir(args.folder)
    if (work / "groups.json").exists():
        photos = scan(args.folder, work, args.jobs)
    else:
        photos = cmd_group(args)
    groups = [g for g in json.loads((work / "groups.json").read_text()) if g]
    unknown = {f for g in groups for f in g} - set(photos)
    if unknown:
        sys.exit(f"groups.json lists photos not in {args.folder}: {', '.join(sorted(unknown))}")
    missing = set(photos) - {f for g in groups for f in g}
    if missing:
        print(f"warning: {len(missing)} photos not in groups.json (run group to regroup)", file=sys.stderr)
    taste = TASTE.read_text() if TASTE.exists() else ""
    ctx = {"folder": args.folder, "work": work, "photos": photos, "model": args.model,
           "taste": taste, "taste_hash": hashlib.sha256(taste.encode()).hexdigest()}

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda g: tournament(ctx, g), groups))

    report = {"model": args.model, "groups": [
        {"id": i, **r, "scores": {f: {k: photos[f][k] for k in ("sharpness", "over", "under", "risks")} for f in g}}
        for i, (g, r) in enumerate(zip(groups, results))
    ]}
    (work / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for g in report["groups"]:
        mark = f"  (review: alt {g['final_alt_best']})" if g["needs_review"] else ""
        print(f"group {g['id']}: {g['best']} of {len(g['ranking'])}{mark}")
    print(f"report: {work / 'report.json'}")


def cmd_override(args):
    report = json.loads((workdir(args.folder) / "report.json").read_text())
    group = report["groups"][args.group]
    if args.file not in group["ranking"]:
        sys.exit(f"{args.file} is not in group {args.group}")
    with OVERRIDES.open("a") as f:
        f.write(json.dumps({
            "time": time.time(), "folder": str(args.folder.resolve()), "group": args.group,
            "ai_best": group["best"], "user_best": args.file, "note": args.note,
        }, ensure_ascii=False) + "\n")
    print(f"recorded in {OVERRIDES}")


LEARN = """Below are cases where a user overrode an AI photo-culling pick. Each attached image shows a pair: 1 = AI pick, 2 = user pick, in the same order as the cases.

Cases:
{cases}

Current taste notes:
{taste}

Rewrite the taste notes as a short markdown list of the user's photo preferences, keeping what still holds and adding what these cases show. Output the full new notes only."""


def cmd_learn(args):
    cases = [json.loads(line) for line in OVERRIDES.read_text().splitlines() if line.strip()]
    img = HERE / "learn"
    img.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, c in enumerate(cases):
        folder = Path(c["folder"])
        out = img / f"pair{i + 1}.jpg"
        grid([folder / c["ai_best"], folder / c["user_best"]], out)
        pairs.append(out)
    text = "\n".join(f"{i + 1}. note: {c['note'] or '(none)'}" for i, c in enumerate(cases))
    taste = TASTE.read_text() if TASTE.exists() else "(none)"
    print(ask_pi(args.model, pairs, LEARN.format(cases=text, taste=taste)))


def main():
    parser = argparse.ArgumentParser(prog="pic_pick.py", description=__doc__)
    sub = parser.add_subparsers(required=True)

    def common(p):
        p.add_argument("folder", type=Path)
        p.add_argument("--gap", type=float, default=2, help="max seconds between photos in a group")
        p.add_argument("--dhash", type=int, default=10, help="max dHash distance (0-64) in a group")
        p.add_argument("--jobs", type=int, default=4, help="parallel workers / pi calls")

    p = sub.add_parser("group", help="group similar photos into groups.json")
    common(p)
    p.set_defaults(func=cmd_group)

    p = sub.add_parser("rank", help="rank each group with the LLM and write report.json")
    common(p)
    p.add_argument("--model", default="gpt-6-luna")
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("override", help="record that you prefer another photo than the AI pick")
    p.add_argument("folder", type=Path)
    p.add_argument("group", type=int)
    p.add_argument("file")
    p.add_argument("--note", default="", help="why you prefer it")
    p.set_defaults(func=cmd_override)

    p = sub.add_parser("learn", help=f"print updated taste notes from overrides (review, then save to {TASTE})")
    p.add_argument("--model", default="gpt-6-luna")
    p.set_defaults(func=cmd_learn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
