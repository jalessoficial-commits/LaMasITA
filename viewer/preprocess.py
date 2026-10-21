"""Turn the ATD, homography and Smart Tagging into compact files the browser viewer can load.

    python viewer/preprocess.py                         # tactical, reads data/ (or MTAH_DATA_DIR)
    python viewer/preprocess.py --source broadcast
    python viewer/preprocess.py --data-dir /elsewhere

Writes into viewer/data/:

    meta_<source>.json          fps, frame count, slot -> team, halves
    atd_<source>.bin            uint16, (n_frames, n_slots + 1, 2): x, y quantised to 0..65000,
                                65535 = NaN; the last object of every frame is the ball
    homography_<source>.bin     float32, (n_frames, 9), NaN where valid == 0
    smart_tagging.json          tactical only, [{start_frame, end_frame, code, team, type}]

Index 0 of every file is video frame 0 (ATD frame_id 1), so the browser never has to think about
the frame_id - 1 shift.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402

# Params
SCALE = 65000
MISSING = 65535
OUT_DIR = os.path.join(HERE, "data")
MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def read_meta(xml_path):
    """Slot ids in channel order, their team side (H/A) and the team names."""
    root = ET.parse(xml_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    side_of_team = {team_id: side for (team_id, _), side in zip(teams, "HA")}
    slots = [(int(player.get("id")), side_of_team.get(player.get("teamId"), "?"))
             for player in root.iter("Player")]
    return slots, {side: name for (_, name), side in zip(teams, "HA")}


def parse_number(text):
    return np.nan if text == "NaN" else float(text)


def convert_atd(raw_path, n_frames, n_slots):
    """Stream the raw text into a quantised (n_frames, n_slots + 1, 2) array."""
    out = np.full((n_frames, n_slots + 1, 2), MISSING, dtype=np.uint16)
    with open(raw_path) as handle:
        for line in handle:
            frame_part, players_part, ball_part = line.rstrip("\n").split(":")
            video_frame = int(frame_part) - 1
            if not 0 <= video_frame < n_frames:
                continue
            values = []
            for slot in players_part.split(";"):
                fields = slot.split(",")
                values.append((parse_number(fields[0]), parse_number(fields[1])))
            ball = ball_part.split(",")
            values.append((parse_number(ball[0]), parse_number(ball[1])))
            row = np.array(values[:n_slots + 1], dtype=np.float64)
            encoded = np.where(np.isnan(row), MISSING,
                               np.clip(np.round(row * SCALE), 0, MISSING - 1))
            out[video_frame, :len(row)] = encoded.astype(np.uint16)
    return out


def convert_homography(csv_path, n_frames):
    table = pd.read_csv(csv_path)
    out = np.full((n_frames, 9), np.nan, dtype=np.float32)
    frames = table["frame"].values
    keep = (frames >= 0) & (frames < n_frames) & (table["valid"].values == 1)
    out[frames[keep]] = table.loc[keep, MATRIX_COLUMNS].values.astype(np.float32)
    return out


def convert_smart_tagging(xml_path, fps):
    events = []
    for instance in ET.parse(xml_path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        events.append({
            "start_frame": int(round(float(instance.findtext("start")) * fps)),
            "end_frame": int(round(float(instance.findtext("end")) * fps)),
            "code": instance.findtext("code"),
            "team": tags.get("Team"),
            "type": tags.get("Type"),
        })
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = asset_paths(GAME, args.data_dir, args.source)
    n_frames = source_frames(args.source, GAME)
    fps = GAMES[GAME]["fps"]

    slots, team_names = read_meta(paths["atd_meta"])
    print("{} slots, teams {}".format(len(slots), team_names))

    meta = {
        "game": GAME,
        "source": args.source,
        "fps": fps,
        "n_frames": n_frames,
        "pitch": GAMES[GAME]["pitch"],
        "scale": SCALE,
        "missing": MISSING,
        "slots": [slot_id for slot_id, _ in slots],
        "slot_teams": [side for _, side in slots],
        "team_names": team_names,
        "halves": half_video_blocks(GAME) if args.source == "tactical" else [],
    }
    with open(os.path.join(OUT_DIR, "meta_{}.json".format(args.source)), "w") as handle:
        json.dump(meta, handle, indent=1)

    print("converting ATD ...")
    atd = convert_atd(paths["atd_raw"], n_frames, len(slots))
    atd.tofile(os.path.join(OUT_DIR, "atd_{}.bin".format(args.source)))

    print("converting homography ...")
    convert_homography(paths["homography"], n_frames).tofile(
        os.path.join(OUT_DIR, "homography_{}.bin".format(args.source)))

    if args.source == "tactical" and os.path.exists(paths.get("smart_tagging", "")):
        with open(os.path.join(OUT_DIR, "smart_tagging.json"), "w") as handle:
            json.dump(convert_smart_tagging(paths["smart_tagging"], fps), handle)

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
