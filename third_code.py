import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


target_name = "זיידנברג נצר יעקב"
folder_path = "/home/ben/Desktop/Final_Project/bridge_data"
threshold = 60

all_results = []    # existing pair-level records
low_hands = []

#per-board records for contract/suit analysis
board_records = []
board_records_field = []  # new for all players

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
    if special_name in name or special_name in name:
        name = name.replace(special_name,"זיידנברגגת אביב לאו")
        name = name.replace(special_name, "זיידנברג גת אביב לאו")
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

def get_declarer_side(cstr):
    """Determine declarer side from contract string"""
    if not cstr:
        return None
    m = re.match(r".*([NESW])", cstr)
    if not m:
        return None
    d = m.group(1)
    return "NS" if d in ("N", "S") else "EW"

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

    # First pass: collect ALL board records for field analysis
    for board in root.iter('board'):
        b_id = board.attrib.get('id')
        for data in board.findall('data'):
            contract_str_all = data.attrib.get("C") or ""
            parsed_all = parse_contract(contract_str_all)
            
            # Determine side for this data entry
            dataN = data.attrib.get('N')
            dataE = data.attrib.get('E')
            side = None
            if dataN:
                side = "NS"
                pair_id_field = dataN
            elif dataE:
                side = "EW"
                pair_id_field = dataE
            else:
                continue
            
            # Add declarer side and defense info for field analysis
            declarer_side = get_declarer_side(contract_str_all)
            is_defense = side != declarer_side if declarer_side else None
            
            board_records_field.append({
                "file": filename,
                "board_id": b_id,
                "pair_id": pair_id_field,
                "side": side,
                "contract_raw": contract_str_all,
                "declarer_side": declarer_side,
                "is_defense": is_defense,
                **(parsed_all or {})
            })

    # Second pass: find target player's pairs and their specific records
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
            partner_name_clean = clean_name(partner_name_raw)  # keep your reversal step for RTL was [::-1]
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
                        
                        # Add declarer side and defense info
                        declarer_side = get_declarer_side(contract_str)
                        is_defense = side != declarer_side if declarer_side else None

                        board_records.append({
                            "file": filename,
                            "board_id": b_id,
                            "pair_id": pair_id,
                            "side": side,
                            "player_pct": player_pct,
                            "board_point": board_point,
                            "contract_raw": contract_str,
                            "declarer_side": declarer_side,
                            "is_defense": is_defense,
                            **(parsed or {})
                        })

# === Build DataFrames ===
df_all = pd.DataFrame(all_results)
df_boards = pd.DataFrame(board_records)
df_boards_field = pd.DataFrame(board_records_field)

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

    print_and_record(f"Average score: {target_name}: {avg:.2f}")#was [::-1]
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
    cols_show = ["partner_name", "partner_gender", "games_played", "avg_score",
                "min_age", "max_age", "min_exp", "max_exp", "min_rank", "max_rank", "weighted_score"]

    # Save as HTML table instead of plain text
    partner_table_html = partner_stats[cols_show].to_html(
        index=False,
        justify="center",
        border=1,
        classes="partner-table"
    )

    report_lines.append(partner_table_html)  # add HTML table directly

    best = partner_stats.iloc[0]
    print_and_record(f"\nBest Partner:\nName: {best['partner_name']}, Gender: {best['partner_gender']}, Weighted Score: {best['weighted_score']:.2f}")

    # PART 2 - Contract vs Result Analysis (Relative)
    print_and_record("\n\n--- Contract vs Result Analysis (Relative) ---")
    if df_boards.empty or df_boards_field.empty:
        print_and_record("No board-level data available.")
    else:
        dfc = df_boards.dropna(subset=["contract_level"]).copy()
        dff = df_boards_field.dropna(subset=["contract_level"]).copy()

        # Average contract level
        avg_level_player = dfc["contract_level"].mean()
        avg_level_field = dff["contract_level"].mean()
        print_and_record(f"Average contract level: Player={avg_level_player:.2f}, Field={avg_level_field:.2f}")

        # Tricks bid vs made
        print_and_record(f"Average tricks bid: Player={dfc['tricks_bid'].mean():.2f}, Field={dff['tricks_bid'].mean():.2f}")
        print_and_record(f"Average tricks made: Player={dfc['tricks_made'].mean():.2f}, Field={dff['tricks_made'].mean():.2f}")

        # Success rates by contract level
        success_by_level_player = dfc.groupby("contract_level")["success"].mean()
        success_by_level_field = dff.groupby("contract_level")["success"].mean()

        # Merge into a DataFrame for plotting
        success_df = pd.DataFrame({
            "Player": success_by_level_player,
            "Field": success_by_level_field
        }).fillna(0)

        print_and_record("\nSuccess rate per contract level:")
        for lvl, row in success_df.iterrows():
            print_and_record(f"Level {lvl}: Player={row['Player']:.2%}, Field={row['Field']:.2%}")

        # === Visualization ===
        plt.figure(figsize=(7,5))
        success_df.plot(kind="bar")
        plt.ylabel("Success Rate")
        plt.title("Success Rate per Contract Level (Player vs Field)")
        plt.xticks(rotation=0)
        plt.legend(title="")
        plot_path = "/home/ben/Desktop/Final_Project/plots/contract_levels.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # Add to HTML
        report_lines.append(f'<img src="{plot_path}" width="600">')

    # PART 3 - Suit Distribution & Play Frequency (FIXED)
    print_and_record("\n\n--- Suit Distribution & Play Frequency ---")
    if df_boards.empty:
        print_and_record("No board-level data to compute suit stats.")
    else:
        dfc = df_boards.dropna(subset=["strain"]).copy()
        dff = df_boards_field.dropna(subset=["strain"]).copy()

        # Frequency of strains
        suit_counts_player = dfc["strain"].value_counts(normalize=True)
        suit_counts_field = dff["strain"].value_counts(normalize=True)

        print_and_record("Suit frequency (player vs field):")
        for suit in ["S","H","D","C","N"]:
            p = suit_counts_player.get(suit,0)
            f = suit_counts_field.get(suit,0)
            print_and_record(f"{suit}: Player={p:.2%}, Field={f:.2%}")

        # Plot suit frequency
        freq_df = pd.DataFrame({"Player": suit_counts_player, "Field": suit_counts_field}).fillna(0)
        
        plt.figure(figsize=(10,5))
        
        # Subplot 1: Frequency
        plt.subplot(1, 2, 1)
        freq_df.plot(kind="bar", ax=plt.gca())
        plt.title("Suit Distribution (Player vs Field)")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        plt.legend(title="")

        # Win rate per suit
        suit_success_player = dfc.groupby("strain")["success"].mean()
        suit_success_field = dff.groupby("strain")["success"].mean()

        succ_df = pd.DataFrame({"Player": suit_success_player, "Field": suit_success_field}).fillna(0)
        
        # Subplot 2: Success rate
        plt.subplot(1, 2, 2)
        succ_df.plot(kind="bar", ax=plt.gca())
        plt.title("Success Rate per Suit (Player vs Field)")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=0)
        plt.legend(title="")

        plt.tight_layout()
        plot_path_suits = "/home/ben/Desktop/Final_Project/plots/suit_analysis.png"
        plt.savefig(plot_path_suits, bbox_inches="tight")
        plt.close()

        # Add to HTML
        report_lines.append(f'<img src="{plot_path_suits}" width="800">')

        print_and_record("\nSuccess rate per suit (player vs field):")
        for suit in ["S","H","D","C","N"]:
            p = suit_success_player.get(suit,0)
            f = suit_success_field.get(suit,0)
            print_and_record(f"{suit}: Player={p:.2%}, Field={f:.2%}")

    # PART 4 - Defense Analysis (ENHANCED WITH FIELD COMPARISON)
    print_and_record("\n\n--- Defense Analysis ---")
    if df_boards.empty:
        print_and_record("No board-level data to compute defense stats.")
    else:
        dfc = df_boards.dropna(subset=["contract_level"]).copy()
        dff = df_boards_field.dropna(subset=["contract_level"]).copy()

        # Filter for valid defense records
        dfc_def = dfc[dfc["is_defense"] == True].copy()
        dff_def = dff[dff["is_defense"] == True].copy()

        # Defense frequency
        total_boards_player = len(dfc)
        defense_boards_player = len(dfc_def)
        total_boards_field = len(dff)
        defense_boards_field = len(dff_def)
        
        defense_freq_player = defense_boards_player / total_boards_player if total_boards_player > 0 else 0
        defense_freq_field = defense_boards_field / total_boards_field if total_boards_field > 0 else 0
        
        print_and_record(f"Defense frequency: Player={defense_freq_player:.2%}, Field={defense_freq_field:.2%}")

        if dfc_def.empty:
            print_and_record("No defense boards found for player.")
        else:
            # Defense success = contract set (over_under < 0) when defending
            success_rate_player = (dfc_def["over_under"] < 0).mean()
            success_rate_field = (dff_def["over_under"] < 0).mean() if not dff_def.empty else 0
            
            print_and_record(f"Defense success rate: Player={success_rate_player:.2%}, Field={success_rate_field:.2%}")

            # Success per strain
            strain_success_player = dfc_def.groupby("strain").apply(lambda x: (x["over_under"] < 0).mean())
            strain_success_field = dff_def.groupby("strain").apply(lambda x: (x["over_under"] < 0).mean()) if not dff_def.empty else pd.Series()

            print_and_record("\nDefense success rate per strain:")
            for strain in ["S","H","D","C","N"]:
                p = strain_success_player.get(strain, 0) if strain in strain_success_player.index else 0
                f = strain_success_field.get(strain, 0) if strain in strain_success_field.index else 0
                print_and_record(f"{strain}: Player={p:.2%}, Field={f:.2%}")

            # Create visualization for defense analysis
            plt.figure(figsize=(12, 4))
            
            # Plot 1: Overall defense stats
            plt.subplot(1, 3, 1)
            categories = ['Defense\nFrequency', 'Defense\nSuccess Rate']
            player_values = [defense_freq_player, success_rate_player]
            field_values = [defense_freq_field, success_rate_field]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, player_values, width, label='Player', alpha=0.8)
            plt.bar(x + width/2, field_values, width, label='Field', alpha=0.8)
            plt.ylabel('Rate')
            plt.title('Defense Overview')
            plt.xticks(x, categories)
            plt.legend()
            plt.ylim(0, 1)
            
            # Plot 2: Defense success by strain
            plt.subplot(1, 3, 2)
            defense_strain_df = pd.DataFrame({
                "Player": strain_success_player,
                "Field": strain_success_field
            }).fillna(0)
            
            if not defense_strain_df.empty:
                defense_strain_df.plot(kind="bar", ax=plt.gca())
                plt.title("Defense Success by Strain")
                plt.ylabel("Success Rate")
                plt.xticks(rotation=0)
                plt.legend(title="")
            
            # Plot 3: Success per strain+level for player only (field might be too sparse)
            plt.subplot(1, 3, 3)
            if not dfc_def.empty:
                dfc_def["strain_level"] = dfc_def["contract_level"].astype(str) + dfc_def["strain"]
                sl_success_player = dfc_def.groupby("strain_level").apply(lambda x: (x["over_under"] < 0).mean())
                
                if len(sl_success_player) > 0:
                    sl_success_player.plot(kind="bar", ax=plt.gca())
                    plt.title("Player Defense Success\nby Contract")
                    plt.ylabel("Success Rate")
                    plt.xticks(rotation=45)
                    
                    print_and_record("\nDefense success rate per strain+level (Player only):")
                    for sl, val in sl_success_player.items():
                        print_and_record(f"{sl}: {val:.2%}")

            plt.tight_layout()
            plot_path_defense = "/home/ben/Desktop/Final_Project/plots/defense_analysis.png"
            plt.savefig(plot_path_defense, bbox_inches="tight")
            plt.close()

            # Add to HTML
            report_lines.append(f'<img src="{plot_path_defense}" width="900">')

# Write the report to an HTML file
safe_name = re.sub(r"[^א-תA-Za-z0-9_\- ]", "", target_name)
out_fname = f"/home/ben/Desktop/Final_Project/player_stats_{safe_name.replace(' ', '_')}.html"

# Convert each line into <p> for readability, wrap in <html>/<body>
html_content = "<html><head><meta charset='utf-8'><title>Player Stats</title></head><body>"
for line in report_lines:
    if line.strip().startswith("---"):
        html_content += f"<h2>{line.strip('- ').strip()}</h2>"
    else:
        html_content += f"<p>{line}</p>"
html_content += "</body></html>"

with open(out_fname, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nHTML report written to {out_fname}")

# Open automatically in browser
import webbrowser
webbrowser.open(f"file://{out_fname}")
