"""AI photo culling: group similar JPEGs, then let Jev pick the best of each group."""

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import typesafe_sdk as typesafe
from PIL import ExifTags, Image, ImageFilter, ImageOps, ImageStat

ISO_MAX = 3200
SIDE = 1568  # max image side the TypeSafe API accepts
MIN_SIDE = 280  # min image side the TypeSafe API accepts
HERE = Path(__file__).resolve().parent


# ---------- step 0: read + step 2: technical scores ----------

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def open_image(path, size):
    im = Image.open(path)
    scale = max(im.size) / size
    im.draft("RGB", (int(im.width / scale), int(im.height / scale)))  # fast JPEG decode at reduced scale
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":  # a gray/CMYK profile would mislabel the converted RGB pixels
        im.info.pop("icc_profile", None)
    return im.convert("RGB")


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
    im = open_image(path, SIDE)
    im.thumbnail((SIDE, SIDE))  # one thumbnail for analysis, Jev, and display
    thumb = work / "thumbs" / f"{sha}.jpg"
    thumb.parent.mkdir(parents=True, exist_ok=True)
    im.save(thumb, quality=90, icc_profile=im.info.get("icc_profile"))
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
    """A photo joins a group only if every member is within max_dist of it (no chaining).
    Candidates are groups with a photo taken within the last gap seconds."""
    dist = lambda a, b: (a["dhash"] ^ b["dhash"]).bit_count()
    groups, group_of, seen = [], {}, []
    for p in sorted(photos.values(), key=lambda p: (p["time"], p["file"])):
        recent = set()
        for q in reversed(seen):
            if p["time"] - q["time"] > gap:
                break
            recent.add(group_of[q["file"]])
        fits = [(max(dist(p, photos[f]) for f in groups[g]), g) for g in recent]
        fits = [x for x in fits if x[0] <= max_dist]
        if fits:
            g = min(fits)[1]
        else:
            g = len(groups)
            groups.append([])
        groups[g].append(p["file"])
        group_of[p["file"]] = g
        seen.append(p)
    return groups


# ---------- step 3: Jev judgments ----------

# One request per photo: Scores for quality dimensions, Nouls for flags. Code combines them.
QUESTIONS = {
    "focus": typesafe.Score(
        instructions="How sharp is the main subject in `photo`?",
        criteria=[
            "The main subject is visibly blurred or out of focus",
            "The main subject is slightly soft",
            "The main subject is crisp and in focus",
        ],
    ),
    "exposure": typesafe.Score(
        instructions="How well exposed is the main subject in `photo`?",
        criteria=[
            "The subject is too dark or too bright and detail is lost",
            "The exposure is a bit off but the subject is clear",
            "The subject is well exposed with detail in highlights and shadows",
        ],
    ),
    "composition": typesafe.Score(
        instructions="How well is `photo` framed?",
        criteria=[
            "The subject is awkwardly placed, tilted, or lost in clutter",
            "The framing is acceptable",
            "The framing leads the eye straight to the subject",
        ],
    ),
    "moment": typesafe.Score(
        instructions="How strong is the moment captured in `photo`?",
        criteria=[
            "Nothing is happening, or people look stiff or caught off guard",
            "An ordinary moment",
            "A peak moment: decisive action, genuine emotion, or interaction",
        ],
    ),
    "closed_eyes": typesafe.Noul(
        instructions="Does a person facing the camera in `photo` have closed or half-closed eyes?",
    ),
    "cropped_limbs": typesafe.Noul(
        instructions="Does the edge of `photo` cut off a person's hands, feet, or the top of the head?",
    ),
    "bad_expression": typesafe.Noul(
        instructions="Does a person in `photo` have an unflattering expression, such as mid-word, grimacing, or mid-blink?",
    ),
}
FLAGS = [q for q, v in QUESTIONS.items() if isinstance(v, typesafe.Noul)]
# composite = weighted Scores (0-1) + relative sharpness - penalty per flag probability
WEIGHTS = {"focus": 0.25, "sharpness": 0.15, "exposure": 0.1, "composition": 0.15, "moment": 0.35}
FLAG_PENALTY = 0.3
REVIEW_GAP = 0.05  # top two composites closer than this -> needs_review


def judge(ctx, photo):
    spec = {q: v.model_dump() for q, v in QUESTIONS.items()}  # editing a question invalidates the cache
    key = hashlib.sha256(json.dumps([photo["sha"], ctx["model"], spec]).encode()).hexdigest()[:16]
    cache = ctx["work"] / "judgments" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    thumb = ctx["work"] / "thumbs" / f"{photo['sha']}.jpg"
    data = thumb.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        if min(im.size) < MIN_SIDE:  # e.g. panoramas
            buf = io.BytesIO()
            padded = ImageOps.pad(im, (max(im.width, MIN_SIDE), max(im.height, MIN_SIDE)))
            padded.save(buf, "JPEG", quality=90, icc_profile=im.info.get("icc_profile"))
            data = buf.getvalue()
    try:
        r = ctx["client"].system_one({"photo": typesafe.Image(data)}, QUESTIONS)
    except typesafe.TypeSafeError as e:
        e.add_note(f"photo: {photo['file']}")
        raise
    result = {
        **{q: a.score / (len(QUESTIONS[q].criteria) - 1) for q, a in r.scores.items()},
        **{q: a.noul for q, a in r.nouls.items()},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


# ---------- step 4: pick ----------

def pick(photos, judgments, group):
    if len(group) == 1:
        return {"best": group[0], "ranking": group, "flags": {}, "needs_review": False}
    top = max(photos[f]["sharpness"] for f in group) or 1
    totals, flags = {}, {}
    for f in group:
        j = {**judgments[f], "sharpness": photos[f]["sharpness"] / top}
        totals[f] = sum(w * j[q] for q, w in WEIGHTS.items()) - FLAG_PENALTY * sum(j[q] for q in FLAGS)
        hits = [q for q in FLAGS if j[q] > 0.5]
        if hits:
            flags[f] = hits
    ranking = sorted(group, key=totals.get, reverse=True)
    return {
        "best": ranking[0],
        "ranking": ranking,
        "flags": flags,
        "needs_review": totals[ranking[0]] - totals[ranking[1]] < REVIEW_GAP,
        "alt_best": ranking[1],
        "totals": totals,
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
    todo = [photos[f] for g in groups if len(g) > 1 for f in g]
    headers = {"TypeSafe-Danger-Zone-Images": "true"}  # images are opt-in
    with typesafe.TypeSafeClient(model=args.model, headers=headers) as client, ThreadPoolExecutor(args.jobs) as pool:
        ctx = {"folder": args.folder, "work": work, "model": args.model, "client": client}
        judgments = dict(zip((p["file"] for p in todo), pool.map(lambda p: judge(ctx, p), todo)))
    results = [pick(photos, judgments, g) for g in groups]

    report = {"model": args.model, "groups": [
        {"id": i, **r, "scores": {
            f: {**{k: photos[f][k] for k in ("sharpness", "over", "under", "risks")}, "jev": judgments.get(f)}
            for f in g
        }}
        for i, (g, r) in enumerate(zip(groups, results))
    ]}
    (work / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for g in report["groups"]:
        mark = f"  (review: alt {g['alt_best']})" if g["needs_review"] else ""
        print(f"group {g['id']}: {g['best']} of {len(g['ranking'])}{mark}")
    print(f"report: {work / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(prog="pic_pick.py", description=__doc__)
    sub = parser.add_subparsers(required=True)

    def common(p):
        p.add_argument("folder", type=Path)
        p.add_argument("--gap", type=float, default=60, help="a photo may join a group with a member taken within this many seconds")
        p.add_argument("--dhash", type=int, default=10, help="max dHash distance (0-64) in a group")
        p.add_argument("--jobs", type=int, default=4, help="parallel workers / Jev calls")

    p = sub.add_parser("group", help="group similar photos into groups.json")
    common(p)
    p.set_defaults(func=cmd_group)

    p = sub.add_parser("rank", help="rank each group with Jev and write report.json")
    common(p)
    p.add_argument("--model", default="jev-latest")
    p.set_defaults(func=cmd_rank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
