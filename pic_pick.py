"""AI photo culling: group similar photos, then let Jev pick the best of each group."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import threading
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import rawpy
import typesafe_sdk as typesafe
from PIL import ExifTags, Image, ImageFilter, ImageOps, ImageStat

ISO_MAX = 3200
SIDE = 1568  # max image side the TypeSafe API accepts
MIN_SIDE = 280  # min image side the TypeSafe API accepts
HERE = Path(__file__).resolve().parent
JPEG_EXTS = {".jpg", ".jpeg"}
RAW_EXTS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".pef", ".raf", ".rw2", ".srw"}
# LibRaw flip -> PIL transpose; ROTATE_90 is counter-clockwise
RAW_FLIP = {3: Image.Transpose.ROTATE_180, 5: Image.Transpose.ROTATE_90, 6: Image.Transpose.ROTATE_270}


class PickError(Exception):
    """A user-facing error: the CLI prints it, the viewer shows it."""


def no_tick(stage, done, total):
    pass


# ---------- step 0: read + step 2: technical scores ----------

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def open_image(src, size, flip=0):
    im = Image.open(src)
    scale = max(im.size) / size
    im.draft("RGB", (int(im.width / scale), int(im.height / scale)))  # fast JPEG decode at reduced scale
    if flip in RAW_FLIP and ExifTags.Base.Orientation not in im.getexif():  # RAW previews often lack the tag
        im = im.transpose(RAW_FLIP[flip])
    else:
        im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":  # a gray/CMYK profile would mislabel the converted RGB pixels
        im.info.pop("icc_profile", None)
    return im.convert("RGB")


def read_exif(im):
    exif = im.getexif()
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    try:
        taken = sub[ExifTags.Base.DateTimeOriginal].strip("\x00 ")
        ts = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S").timestamp()
    except (KeyError, ValueError):  # missing or unset camera clock, e.g. 0000:00:00
        ts = None
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


def load(path):
    """Upright RGB image reduced to about SIDE, plus capture info. RAW uses its embedded JPEG preview."""
    if path.suffix.lower() in JPEG_EXTS:
        data, flip = path.read_bytes(), 0
        with Image.open(path) as im:
            exif = read_exif(im)
    else:
        with rawpy.imread(str(path)) as raw:
            data, flip = raw_preview(raw, path), raw.sizes.flip
            with Image.open(io.BytesIO(data)) as im:
                exif = read_exif(im)
            if exif["time"] is None:  # many previews carry no EXIF; .other unpacks the whole raw
                try:
                    o = raw.other
                except rawpy.LibRawNonFatalError:  # e.g. unsupported sensor data; preview still usable
                    pass
                else:
                    exif = {
                        "time": o.timestamp.timestamp() if o.timestamp.year > 1970 else None,
                        "shutter": o.shutter_speed or None,
                        "focal": o.focal_length or None,  # actual, not 35 mm equivalent
                        "iso": int(o.iso_speed) or None,
                    }
    if exif["time"] is None:
        exif["time"] = path.stat().st_mtime
    return open_image(io.BytesIO(data), SIDE, flip), exif


def raw_preview(raw, path):
    thumb = raw.extract_thumb()
    if thumb.format != rawpy.ThumbFormat.JPEG:
        raise ValueError(f"{path.name}: no embedded JPEG preview")
    return thumb.data


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
    im, exif = load(path)
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
        "color": "#%02x%02x%02x" % im.resize((1, 1)).getpixel((0, 0)),  # viewer placeholder
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(info))
    return {**info, "file": path.name}


def picked_dir(folder):  # folder is resolved by argparse, so "." works
    return folder.parent / f"{folder.name}-picked"


def find_photos(folder):
    """{photo name: [paths]} across folder and its -picked sibling. Files sharing a stem
    (RAW+JPEG) are one photo, named after the JPEG when there is one. Other files starting
    with the stem plus a dot, such as IMG_1.xmp or IMG_1.CR3.xmp sidecars, come along."""
    by_stem, others = {}, []
    for d in (folder, picked_dir(folder)):
        for p in sorted(d.iterdir()) if d.is_dir() else []:
            if p.suffix.lower() in JPEG_EXTS | RAW_EXTS:
                by_stem.setdefault(p.stem, []).append(p)
            elif p.is_file():
                others.append(p)
    for stem, paths in by_stem.items():
        if len({p.parent for p in paths}) > 1:
            raise PickError(f"files of {stem} are split between {folder} and {picked_dir(folder)}; move them into one folder")
        paths.sort(key=lambda p: p.suffix.lower() not in JPEG_EXTS)  # JPEG first: it names the photo
    for p in others:  # a sidecar belongs to the longest matching stem, e.g. IMG_1.edit.xmp -> IMG_1.edit
        parts = p.name.split(".")
        stems = [s for i in range(len(parts) - 1, 0, -1)
                 if (s := ".".join(parts[:i])) in by_stem and by_stem[s][0].parent == p.parent]
        if stems:
            by_stem[stems[0]].append(p)
    return {paths[0].name: paths for paths in by_stem.values()}


def scan(folder, work, jobs, tick=no_tick):
    found = find_photos(folder)
    work.mkdir(parents=True, exist_ok=True)
    done = count(1)

    def one(paths):
        info = analyze(paths[0], work)
        tick("scan", next(done), len(found))
        return info

    with ThreadPoolExecutor(jobs) as pool:
        photos = {p["file"]: p for p in pool.map(one, found.values())}
    (work / "index.json").write_text(json.dumps(photos))  # lets the viewer map names to thumbnails
    return photos


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
    photos = scan(args.folder, work, args.jobs, args.tick)
    groups = make_groups(photos, args.gap, args.dhash)
    (work / "groups.json").write_text(json.dumps(groups, indent=1, ensure_ascii=False))
    print(f"{len(photos)} photos -> {len(groups)} groups: {work / 'groups.json'}")
    print("Edit that file to split or merge groups, then run: uv run pic_pick.py rank", args.folder)
    return photos


def cmd_rank(args):
    work = workdir(args.folder)
    if (work / "groups.json").exists():
        photos = scan(args.folder, work, args.jobs, args.tick)
    else:
        photos = cmd_group(args)
    groups = [g for g in json.loads((work / "groups.json").read_text()) if g]
    unknown = {f for g in groups for f in g} - set(photos)
    if unknown:
        raise PickError(f"groups.json lists photos not in {args.folder}: {', '.join(sorted(unknown))}")
    missing = set(photos) - {f for g in groups for f in g}
    if missing:
        print(f"warning: {len(missing)} photos not in groups.json (run group to regroup)", file=sys.stderr)
    todo = [photos[f] for g in groups if len(g) > 1 for f in g]
    headers = {"TypeSafe-Danger-Zone-Images": "true"}  # images are opt-in
    with typesafe.TypeSafeClient(model=args.model, headers=headers) as client, ThreadPoolExecutor(args.jobs) as pool:
        ctx = {"folder": args.folder, "work": work, "model": args.model, "client": client}
        done = count(1)

        def one(p):
            j = judge(ctx, p)
            args.tick("judge", next(done), len(todo))
            return j

        judgments = dict(zip((p["file"] for p in todo), pool.map(one, todo)))
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


def cmd_export(args):
    """Move kept photos (all files sharing the stem) to <folder>-picked and the rest back.
    Kept = each group's best, plus picks.json "keep", minus picks.json "drop"."""
    work = workdir(args.folder)
    groups, ranked = ranked_groups(work)
    if not ranked:
        raise PickError(f"no report.json in {work}; run rank first")
    if any(frozenset(g) not in ranked for g in groups):
        raise PickError("groups changed since the last rank; run rank again")
    best = {ranked[frozenset(g)]["best"] for g in groups}
    picks = json.loads((work / "picks.json").read_text()) if (work / "picks.json").exists() else {}
    keep = (best | set(picks.get("keep", []))) - set(picks.get("drop", []))
    dest_keep = picked_dir(args.folder)
    found = find_photos(args.folder)
    unknown = keep - set(found)
    if unknown:
        raise PickError(f"kept photos not found: {', '.join(sorted(unknown))}")
    moves = [
        (p, dest / p.name)
        for name, paths in found.items()
        for dest in [dest_keep if name in keep else args.folder]
        for p in paths if p.parent != dest
    ]
    clash = [str(t) for _, t in moves if t.exists()]  # rename would overwrite, e.g. case-only clash on macOS
    if clash:
        raise PickError(f"would overwrite: {', '.join(clash)}")
    for p, t in moves:
        t.parent.mkdir(exist_ok=True)
        p.rename(t)
    message = f"moved {len(moves)} files; kept photos are in {dest_keep}"
    print(message)
    return message


def ranked_groups(work):
    """groups.json, and report.json groups keyed by member set. A regroup after rank leaves
    some groups unranked; export and the viewer both use only the matching ones."""
    read = lambda name: json.loads((work / name).read_text()) if (work / name).exists() else None
    groups, report = read("groups.json"), read("report.json")
    ranked = {frozenset(r["ranking"]): r for r in report["groups"]} if report else {}
    return [g for g in groups or [] if g], ranked


# ---------- viewer ----------

JOB = {"kind": None, "running": False, "stage": "", "done": 0, "total": 0, "error": None, "message": None}
JOB_LOCK = threading.Lock()
PICK_DIALOG = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
print(filedialog.askdirectory(title="pic-pick: choose a photo folder") or "")
"""


def start_job(kind, folder):
    """Run group/rank/export in the background; the viewer polls JOB for progress."""
    with JOB_LOCK:
        if JOB["running"]:
            raise PickError(f"{JOB['kind']} is still running")
        JOB.update(kind=kind, running=True, stage="", done=0, total=0, error=None, message=None)
    args = build_parser().parse_args([kind, str(folder)])  # CLI defaults
    args.tick = lambda stage, done, total: JOB.update(stage=stage, done=done, total=total)

    def run():
        try:
            result = args.func(args)
            JOB["message"] = result if isinstance(result, str) else None
        except Exception as e:
            traceback.print_exc()
            JOB["error"] = str(e) or type(e).__name__
        finally:
            JOB["running"] = False

    # Ctrl-C waits for an export to finish so a photo's files are never left split
    threading.Thread(target=run, daemon=kind != "export").start()


def view_state(folder):
    """Groups with thumbnails, Jev results when ranked, and the user's picks."""
    work = workdir(folder)
    read = lambda name: json.loads((work / name).read_text()) if (work / name).exists() else None
    if not (work / "groups.json").exists():
        return {"groups": None, "keep": [], "drop": [], "seen": []}
    groups, ranked = ranked_groups(work)
    index, picks = read("index.json"), read("picks.json")
    out = []
    for g in groups:
        r = ranked.get(frozenset(g))
        members = [{
            "name": f, "sha": index[f]["sha"], "color": index[f]["color"],
            "total": r and r.get("totals", {}).get(f), "flags": r and r["flags"].get(f, []),
            "scores": r and r["scores"][f],
        } for f in (r["ranking"] if r else g) if f in index]
        if members:
            out.append({"members": members, "best": r and r["best"], "needs_review": bool(r and r["needs_review"])})
    return {"groups": out, **{k: (picks or {}).get(k, []) for k in ("keep", "drop", "seen")}}


def full_image(folder, name):
    """Original JPEG bytes (browsers apply EXIF orientation) or an upright RAW preview."""
    photos = find_photos(folder)
    if name not in photos:
        raise PickError(f"unknown photo {name}")
    path = photos[name][0]
    if path.suffix.lower() in JPEG_EXTS:
        return path.read_bytes()
    with rawpy.imread(str(path)) as raw:
        data, flip = raw_preview(raw, path), raw.sizes.flip
    im = open_image(io.BytesIO(data), 1 << 16, flip)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, icc_profile=im.info.get("icc_profile"))
    return buf.getvalue()


class Viewer(BaseHTTPRequestHandler):
    def log_message(self, *args):  # thumbnail requests would flood the terminal
        pass

    def send(self, body, ctype="application/json", status=200, headers=()):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Frame-Options", "DENY")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def handle_any(self, route):
        # Only this page may call the API: the Host check stops DNS rebinding, and requiring a
        # JSON POST body forces a CORS preflight that other sites cannot pass.
        if self.headers.get("Host") not in self.server.hosts:
            return self.send({"error": "bad host"}, status=403)
        # GETs too: other sites could probe files via <img> or burn CPU on RAW decodes
        if self.headers.get("Sec-Fetch-Site", "none") not in ("same-origin", "none"):
            return self.send({"error": "cross-site request"}, status=403)
        try:
            url = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if self.command == "POST":
                if self.headers.get("Content-Type") != "application/json":
                    return self.send({"error": "JSON only"}, status=415)
                query = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            if "folder" in query:
                query["folder"] = Path(query["folder"]).expanduser().resolve()
                if not query["folder"].is_dir():
                    raise PickError(f"not a folder: {query['folder']}")
            route(url.path, query)
        except PickError as e:
            self.send({"error": str(e)}, status=400)
        except Exception as e:
            traceback.print_exc()
            self.send({"error": str(e) or type(e).__name__}, status=500)

    def do_GET(self):
        self.handle_any(self.get)

    def do_POST(self):
        self.handle_any(self.post)

    def get(self, path, q):
        if path == "/":
            self.send((HERE / "viewer.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send(view_state(q["folder"]))
        elif path == "/api/job":
            self.send(JOB)
        elif path == "/api/thumb":
            sha = q["sha"]
            if not (len(sha) == 16 and all(c in "0123456789abcdef" for c in sha)):
                raise PickError("bad sha")
            data = (workdir(q["folder"]) / "thumbs" / f"{sha}.jpg").read_bytes()
            self.send(data, "image/jpeg", headers=[("Cache-Control", "max-age=31536000, immutable")])
        elif path == "/api/full":
            self.send(full_image(q["folder"], q["name"]), "image/jpeg")
        else:
            self.send({"error": "not found"}, status=404)

    def post(self, path, q):
        if path == "/api/choose":  # tkinter needs the main thread on macOS, so run it as its own process
            r = subprocess.run([sys.executable, "-c", PICK_DIALOG], capture_output=True, text=True)
            if r.returncode:
                raise PickError("folder dialog failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
            self.send({"folder": r.stdout.strip()})
        elif path == "/api/picks":
            work = workdir(q["folder"])
            work.mkdir(parents=True, exist_ok=True)
            (work / "picks.json").write_text(json.dumps({k: q[k] for k in ("keep", "drop", "seen")}, ensure_ascii=False))
            self.send({})
        elif path in ("/api/group", "/api/rank", "/api/export"):
            start_job(path.removeprefix("/api/"), q["folder"])
            self.send({})
        else:
            self.send({"error": "not found"}, status=404)


def cmd_serve(args):
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Viewer)
    server.hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
    url = f"http://127.0.0.1:{args.port}/"
    print(f"viewer: {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def folder_arg(s):
    return Path(s).resolve()


def build_parser():
    parser = argparse.ArgumentParser(prog="pic_pick.py", description=__doc__)
    sub = parser.add_subparsers(required=True)

    def common(p):
        p.add_argument("folder", type=folder_arg)
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

    p = sub.add_parser("export", help="move kept photos (each group's best, adjusted by picks.json) to <folder>-picked")
    p.add_argument("folder", type=folder_arg)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("serve", help="open the browser viewer")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)
    return parser


def main():
    args = build_parser().parse_args()
    args.tick = no_tick
    try:
        args.func(args)
    except PickError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
