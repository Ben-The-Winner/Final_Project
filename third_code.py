import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
from collections import Counter, defaultdict

target_name = "זיידנברג נצר יעקב"
folder_path = "/home/ben/Desktop/Final_Project/bridge_data"
threshold = 60

all_results = []    # existing pair-level records
low_hands = []

# New: per-board records for contract/suit analysis
board_records = []

def clean_name(name):
    # Keep only Hebrew letters and spaces
    name = re.sub(r"[^א-ת\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def normalize_partner_name(name):
    name = re.sub(r"\s+", " ", name).strip()
    return name

def fix_special_name_cases(name):
    special_name = "זיידנברגגת אביב לאו"
    # handle forward or reversed forms
    if special_name[::-1] in name or special_name in name:
        name = name.replace(special_name[::-1], "ואל ביבא תג גרבנדייז")
        name = name.replace(special_name, "ואל ביבא תג גרבנדייז")
    return name

def normalize_hebrew(s):
    if not s:
        return ""
    s = re.sub(r"[^א-ת\s]", "", s)  # Remove non-Hebrew characters
    s = re.sub(r"\s+", " ", s)      # Collapse spaces
    return s.strip()

target_name_norm = normalize_hebrew(target_name)

def parse_contract(contract_str):
    """
    Parse contract like '4SN+1', '3H=', '5DXE-2', '4SXN=' into components.
    Returns dict or None.
    """
    if not contract_str:
        return None
    # Remove unexpected whitespace
    cs = contract_str.strip()

    # attempt regex: level, strain, optional doubling (X/XX), optional declarer letters, result (= or +n or -n)
    # Example forms in your files: 4SN=, 3SN+1, 5DE-2, 4SXN=
    m = re.match(r"^(\d)([SHDCN])(?:X{1,2})?[A-Z]*([=+-]\d+|=)?$", cs)
    if not m:
        # fallback: try to grab level/strain then any trailing =/+/- digits
        m2 = re.match(r"^(\d)([SHDCN]).*?([=+-]\d+|=)?$", cs)
        if not m2:
            return None
        m = m2

    level = int(m.group(1))
    strain = m.group(2)
    result_str = m.group(3) or "="

    tricks_bid = level + 6
    if result_str == "=" or result_str == "= ":
        tricks_made = tricks_bid
    elif result_str.startswith('+'):
        try:
            n = int(result_str[1:])
            tricks_made = tricks_bid + n
        except:
            tricks_made = tricks_bid
    elif result_str.startswith('-'):
        try:
            n = int(result_str[1:])
            tricks_made = tricks_bid - n
        except:
            tricks_made = tricks_bid
    else:
        tricks_made = tricks_bid

    over_under = tricks_made - tricks_bid  # positive = overtricks, 0 exact, negative down
    success = over_under >= 0

    return {
        "contract_level": level,
        "strain": strain,
        "tricks_bid": tricks_bid,
        "tricks_made": tricks_made,
        "over_under": over_under,
        "success": success,
        "raw": contract_str
    }

# Walk all XML files and collect pair-level and board-level info
for filename in os.listdir(folder_path):
    if not filename.endswith(".xml"):
        continue
    filepath = os.path.join(folder_path, filename)
    try:
        tree = ET.parse(filepath)
    except Exception as e:
        print(f"Failed parsing {filename}: {e}")
        continue
    root = tree.getroot()

    # We'll need to find every pair where target appears and also note that pair's id so we can match boards
    for pair in root.iter('pair'):
        names_raw = pair.find('names').text or ""
        names_norm = normalize_hebrew(names_raw)

        if target_name_norm in names_norm:
            # Found the player in this tournament / file
            pair_id = pair.attrib.get('id')
            # restot is pair tournament total (used earlier)
            restot_tag = pair.find('restot')
            score = float(restot_tag.text) if restot_tag is not None and restot_tag.text else np.nan

            ibfn1_tag = pair.find("ibfn1")
            ibfn2_tag = pair.find("ibfn2")
            ibfn1_val = ibfn1_tag.text.strip() if ibfn1_tag is not None and ibfn1_tag.text else ""
            ibfn2_val = ibfn2_tag.text.strip() if ibfn2_tag is not None and ibfn2_tag.text else ""

            # split the names field by '-' (typical separator)
            parts = [p.strip() for p in names_raw.split("-")]
            parts_norm = [normalize_hebrew(p) for p in parts]

            # Determine which side in the 'names' string corresponds to target and partner
            partner_tag = None
            partner_name_clean = ""
            if len(parts_norm) == 2:
                if parts_norm[0] == target_name_norm:
                    partner_name_raw = parts[1]
                    partner_tag = ibfn2_tag
                elif parts_norm[1] == target_name_norm:
                    partner_name_raw = parts[0]
                    partner_tag = ibfn1_tag
                else:
                    # fallback: try substring remove
                    partner_name_raw = names_raw.replace(target_name, "").replace("-", "").strip()
                    # choose partner_tag as the ibfn that is not empty
                    partner_tag = ibfn1_tag if ibfn1_tag is not None and (ibfn1_val != "") else ibfn2_tag
            else:
                # fallback: remove the target name from whole string
                partner_name_raw = names_raw.replace(target_name, "").replace("-", "").strip()
                partner_tag = ibfn1_tag if ibfn1_tag is not None and (ibfn1_val != "") else ibfn2_tag

            # Clean partner name
            partner_name_clean = clean_name(partner_name_raw[::-1])  # keep your reversal step for RTL
            partner_name_clean = fix_special_name_cases(partner_name_clean)
            partner_name_clean = normalize_partner_name(partner_name_clean)

            # partner attributes
            partner_gender = partner_tag.attrib.get("G") if partner_tag is not None else None
            if partner_gender:
                partner_gender = partner_gender.upper()
            else:
                partner_gender = "Unknown"
            try:
                partner_age = int(partner_tag.attrib.get("A")) if partner_tag is not None and partner_tag.attrib.get("A") else 0
            except:
                partner_age = 0
            try:
                partner_exp = int(partner_tag.attrib.get("E")) if partner_tag is not None and partner_tag.attrib.get("E") else 0
            except:
                partner_exp = 0
            partner_rank_raw = partner_tag.attrib.get("R") if partner_tag is not None else None
            try:
                partner_rank = int(partner_rank_raw) if partner_rank_raw is not None else None
            except:
                partner_rank = None

            carry = 0.0
            carry_tag = pair.find('carry')
            if carry_tag is not None and carry_tag.text:
                try:
                    carry = float(carry_tag.text)
                except:
                    carry = 0.0
            nmp = 0
            nmp_tag = pair.find('nmp')
            if nmp_tag is not None and nmp_tag.text:
                try:
                    nmp = int(nmp_tag.text)
                except:
                    nmp = 0

            # Save pair-level info to all_results (same as before)
            all_results.append({
                "file": filename,
                "pair_id": pair_id,
                "score": score,
                "partner_gender": partner_gender,
                "partner_age": partner_age,
                "partner_experience": partner_exp,
                "partner_rank": partner_rank,
                "carry": carry,
                "nmp": nmp,
                "partner_name": partner_name_clean
            })
            if score < threshold:
                low_hands.append((filename, pair_id, score))

            # Now find boards in this same file for this pair (match pair_id to data@N or data@E)
            # iterate boards:
            for board in root.iter('board'):
                b_id = board.attrib.get('id')
                for data in board.findall('data'):
                    # data has attributes like N, E, nss, ews, ns, ew, C (contract), L (table), etc.
                    # Determine if this data row corresponds to our pair (N or E equals the pair id)
                    dataN = data.attrib.get('N')
                    dataE = data.attrib.get('E')
                    if dataN == pair_id or dataE == pair_id:
                        # Determine side and player score percent for comparison
                        side = "NS" if dataN == pair_id else "EW"
                        # player's percent for that row
                        player_pct = None
                        try:
                            if side == "NS":
                                player_pct = float(data.attrib.get("nss", 0))
                            else:
                                player_pct = float(data.attrib.get("ews", 0))
                        except:
                            # fallback if fields missing
                            player_pct = None

                        # absolute score (ns/ew) may be in 'ns' or 'ew' attributes (raw IMP/points)
                        board_point = None
                        try:
                            board_point = float(data.attrib.get("ns")) if side == "NS" else float(data.attrib.get("ew"))
                        except:
                            board_point = None

                        # contract string
                        contract_str = data.attrib.get("C") or ""
                        parsed = parse_contract(contract_str)

                        board_records.append({
                            "file": filename,
                            "board_id": b_id,
                            "pair_id": pair_id,
                            "side": side,
                            "player_pct": player_pct,
                            "board_point": board_point,
                            "contract_raw": contract_str,
                            **(parsed or {})
                        })

# === Build DataFrames ===
df_all = pd.DataFrame(all_results)
df_boards = pd.DataFrame(board_records)

# === Section outputs & file report ===
report_lines = []
def print_and_record(s=""):
    print(s)
    report_lines.append(s)

if df_all.empty:
    print(f"Couldn't find this player: {target_name}")
else:
    # PART 1 - Partner statistics (as before)
    avg = df_all["score"].mean()
    std_dev = df_all["score"].std()

    print_and_record(f"Average score: {target_name[::-1]}: {avg:.2f}")
    print_and_record(f"Standard deviation: {std_dev:.2f}")
    print_and_record("\nHands with a score less than {thr}:".format(thr=threshold))
    for fname, pid, score in low_hands:
        print_and_record(f"file: {fname}, hand: {pid}, score: {score}")

    partner_stats = df_all.groupby(["partner_name", "partner_gender"]).agg(
        games_played=("score", "count"),
        avg_score=("score", "mean"),
        min_age=("partner_age", "min"),
        max_age=("partner_age", "max"),
        min_exp=("partner_experience", "min"),
        max_exp=("partner_experience", "max"),
        min_rank=("partner_rank", lambda x: min([v for v in x if v is not None]) if any(v is not None for v in x) else None),
        max_rank=("partner_rank", lambda x: max([v for v in x if v is not None]) if any(v is not None for v in x) else None)
    ).reset_index()

    # Weighted score - keep your normalization scheme
    partner_stats["avg_score_norm"] = (partner_stats["avg_score"] - partner_stats["avg_score"].min()) / (
        partner_stats["avg_score"].max() - partner_stats["avg_score"].min() + 1e-6)
    partner_stats["games_played_norm"] = (partner_stats["games_played"] - partner_stats["games_played"].min()) / (
        partner_stats["games_played"].max() - partner_stats["games_played"].min() + 1e-6)
    weight_score = 0.7
    weight_games = 0.3
    partner_stats["weighted_score"] = (partner_stats["avg_score_norm"] * weight_score +
                                       partner_stats["games_played_norm"] * weight_games) * 50

    partner_stats = partner_stats.sort_values(by="weighted_score", ascending=False).reset_index(drop=True)

    print_and_record("\n--- Partner Statistics (Weighted Ranking) ---")
    # print subset columns
    cols_show = ["partner_name", "partner_gender", "games_played", "avg_score",
                 "min_age", "max_age", "min_exp", "max_exp", "min_rank", "max_rank", "weighted_score"]
    print_and_record(partner_stats[cols_show].to_string(index=False))

    best = partner_stats.iloc[0]
    print_and_record(f"\nBest Partner:\nName: {best['partner_name']}, Gender: {best['partner_gender']}, Weighted Score: {best['weighted_score']:.2f}")

    # PART 2 - Contract vs Result Analysis
    print_and_record("\n\n--- Contract vs Result Analysis ---")
    if df_boards.empty:
        print_and_record("No board-level contract data found for this player.")
    else:
        # drop rows where contract parsing failed (None contract_level)
        dfc = df_boards.dropna(subset=["contract_level"]).copy()

        # Average contract level announced
        avg_level = dfc["contract_level"].mean()
        print_and_record(f"Average contract level announced: {avg_level:.2f}")

        # Average tricks bid vs average tricks made
        avg_bid = dfc["tricks_bid"].mean()
        avg_made = dfc["tricks_made"].mean()
        print_and_record(f"Average tricks bid: {avg_bid:.2f}")
        print_and_record(f"Average tricks made: {avg_made:.2f}")
        print_and_record(f"Average over/under (made - bid): {dfc['over_under'].mean():.2f}")

        # Frequency exact vs down vs over
        exact_count = (dfc["over_under"] == 0).sum()
        over_count = (dfc["over_under"] > 0).sum()
        down_count = (dfc["over_under"] < 0).sum()
        total_contracts = len(dfc)
        print_and_record(f"\nTotal contracts parsed: {total_contracts}")
        print_and_record(f"Exact (made): {exact_count} ({exact_count/total_contracts:.2%})")
        print_and_record(f"Overtricks: {over_count} ({over_count/total_contracts:.2%})")
        print_and_record(f"Down (failed): {down_count} ({down_count/total_contracts:.2%})")

        # Success rate per contract level
        success_by_level = dfc.groupby("contract_level")["success"].agg(["count", "mean"]).reset_index().sort_values("contract_level")
        print_and_record("\nSuccess rate per contract level (count, success_rate):")
        for _, row in success_by_level.iterrows():
            print_and_record(f"Level {int(row['contract_level'])}: count={int(row['count'])}, success={row['mean']:.2%}")

        # Optionally: success per strain+level (e.g., 3NT)
        dfc["strain_level"] = dfc["strain"] + dfc["contract_level"].astype(int).astype(str)
        success_by_strain_level = dfc.groupby("strain_level")["success"].agg(["count", "mean"]).reset_index()
        print_and_record("\nSuccess by strain+level (e.g., S4):")
        for _, row in success_by_strain_level.iterrows():
            print_and_record(f"{row['strain_level']}: count={int(row['count'])}, success={row['mean']:.2%}")

    # PART 3 - Suit Distribution & Play Frequency
    print_and_record("\n\n--- Suit Distribution & Play Frequency ---")
    if df_boards.empty:
        print_and_record("No board-level data to compute suit stats.")
    else:
        dfc = df_boards.dropna(subset=["strain"]).copy()

        # Frequency of strains
        suit_counts = dfc["strain"].value_counts()
        suit_perc = suit_counts / suit_counts.sum()
        print_and_record("Suit frequency (counts):")
        for s, c in suit_counts.items():
            print_and_record(f"{s}: {c} ({suit_perc[s]:.2%})")

        # Preferred strain
        preferred = suit_counts.idxmax() if not suit_counts.empty else None
        print_and_record(f"\nPreferred strain: {preferred}")

        # Win rate per suit: average player_pct or average success
        # Here we show both: average percent score and success rate for contracts in that strain
        suit_avg_pct = dfc.groupby("strain")["player_pct"].mean()
        suit_success = dfc.groupby("strain")["success"].mean()
        print_and_record("\nAverage player percent (nss/ews) per strain:")
        for s, val in suit_avg_pct.items():
            print_and_record(f"{s}: {val:.2f}")

        print_and_record("\nSuccess rate per strain:")
        for s, val in suit_success.items():
            print_and_record(f"{s}: {val:.2%}")

    # Write the report to a text file
    safe_name = re.sub(r"[^א-תA-Za-z0-9_\- ]", "", target_name)
    out_fname = f"/home/ben/Desktop/Final_Project/player_stats_{safe_name.replace(' ', '_')}.txt"
    with open(out_fname, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport written to {out_fname}")
