import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# === CONFIGURATION ===
target_name = "זיידנברג נצר יעקב"
folder_path = "/home/ben/Desktop/Final_Project/bridge_data"
threshold = 60
plot_dir = "/home/ben/Desktop/Final_Project/plots"
os.makedirs(plot_dir, exist_ok=True)

# === UTILITIES ===
def normalize_hebrew(s):
    if not s:
        return ""
    s = re.sub(r"[^א-ת\s]", "", s)  # Remove non-Hebrew characters
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_contract(contract_str):
    if not contract_str:
        return None
    cs = contract_str.strip()
    m = re.match(r"^(\d)([SHDCN])(?:X{1,2})?[A-Z]*([=+-]\d+|=)?$", cs)
    if not m:
        m2 = re.match(r"^(\d)([SHDCN]).*?([=+-]\d+|=)?$", cs)
        if not m2:
            return None
        m = m2
    level = int(m.group(1))
    strain = m.group(2)
    result_str = m.group(3) or "="
    tricks_bid = level + 6
    if result_str.startswith("+"):
        tricks_made = tricks_bid + int(result_str[1:])
    elif result_str.startswith("-"):
        tricks_made = tricks_bid - int(result_str[1:])
    else:
        tricks_made = tricks_bid
    return {
        "contract_level": level,
        "strain": strain,
        "tricks_bid": tricks_bid,
        "tricks_made": tricks_made,
        "over_under": tricks_made - tricks_bid,
        "success": tricks_made >= tricks_bid,
        "raw": contract_str
    }

def get_declarer_side(cstr):
    if not cstr:
        return None
    m = re.match(r".*([NESW])", cstr)
    if not m:
        return None
    d = m.group(1)
    return "NS" if d in ("N", "S") else "EW"

# === DATA EXTRACTION ===
all_results = []
board_records = []
board_records_field = []

target_name_norm = normalize_hebrew(target_name)

for filename in os.listdir(folder_path):
    if not filename.endswith(".xml"):
        continue
    filepath = os.path.join(folder_path, filename)
    try:
        tree = ET.parse(filepath)
    except:
        continue
    root = tree.getroot()

    # collect board-level field data
    for board in root.iter('board'):
        b_id = board.attrib.get('id')
        for data in board.findall('data'):
            contract_str_all = data.attrib.get("C") or ""
            parsed_all = parse_contract(contract_str_all)
            side = "NS" if data.attrib.get('N') else "EW" if data.attrib.get('E') else None
            if not side:
                continue
            declarer_side = get_declarer_side(contract_str_all)
            is_defense = side != declarer_side if declarer_side else None
            board_records_field.append({
                "file": filename,
                "board_id": b_id,
                "side": side,
                "contract_raw": contract_str_all,
                "declarer_side": declarer_side,
                "is_defense": is_defense,
                **(parsed_all or {})
            })

    # find target player
    for pair in root.iter('pair'):
        names_raw = pair.find('names').text or ""
        names_norm = normalize_hebrew(names_raw)
        if target_name_norm in names_norm:
            pair_id = pair.attrib.get('id')
            score_tag = pair.find('restot')
            score = float(score_tag.text) if score_tag is not None and score_tag.text else np.nan
            all_results.append({"file": filename, "pair_id": pair_id, "score": score})
            # boards of this pair
            for board in root.iter('board'):
                b_id = board.attrib.get('id')
                for data in board.findall('data'):
                    if data.attrib.get('N') == pair_id or data.attrib.get('E') == pair_id:
                        side = "NS" if data.attrib.get('N') == pair_id else "EW"
                        player_pct = float(data.attrib.get("nss" if side=="NS" else "ews", 0))
                        contract_str = data.attrib.get("C") or ""
                        parsed = parse_contract(contract_str)
                        declarer_side = get_declarer_side(contract_str)
                        is_defense = side != declarer_side if declarer_side else None
                        board_records.append({
                            "file": filename,
                            "board_id": b_id,
                            "pair_id": pair_id,
                            "side": side,
                            "player_pct": player_pct,
                            "contract_raw": contract_str,
                            "declarer_side": declarer_side,
                            "is_defense": is_defense,
                            **(parsed or {})
                        })

# === DATAFRAMES ===
df_all = pd.DataFrame(all_results)
df_boards = pd.DataFrame(board_records)
df_boards_field = pd.DataFrame(board_records_field)

# === REPORT ===
report_lines = []
def print_and_record(s=""):
    print(s)
    report_lines.append(s)

def plot_comparison(categories, player_vals, field_vals, title, filename, ylabel=""):
    x = np.arange(len(categories))
    width = 0.35
    plt.figure(figsize=(7,5))
    plt.bar(x - width/2, player_vals, width, label="Player")
    plt.bar(x + width/2, field_vals, width, label="Field")
    plt.xticks(x, categories)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    path = os.path.join(plot_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

if df_all.empty:
    print(f"Couldn't find {target_name}")
else:
    # Bidding Aggressiveness
    print_and_record("\n--- Bidding Aggressiveness ---")
    avg_level_player = df_boards["contract_level"].mean()
    avg_level_field = df_boards_field["contract_level"].mean()
    print_and_record(f"Average contract level: Player={avg_level_player:.2f}, Field={avg_level_field:.2f}")
    path = plot_comparison(["Contract Level"],[avg_level_player],[avg_level_field],"Bidding Aggressiveness","bidding_aggressiveness.png","Avg Level")
    report_lines.append(f'<img src="{path}" width="600">')

    # Overbid vs Underbid - FIXED: Only consider declarer contracts
    print_and_record("\n--- Overbid vs Underbid (As Declarer Only) ---")
    # Filter for declarer contracts only
    player_declarer = df_boards[df_boards["is_defense"] == False]
    field_declarer = df_boards_field[df_boards_field["is_defense"] == False]
    
    if not player_declarer.empty and not field_declarer.empty:
        fail_rate_player = (player_declarer["success"] == False).mean()
        make_rate_player = (player_declarer["success"] == True).mean()
        fail_rate_field = (field_declarer["success"] == False).mean()
        make_rate_field = (field_declarer["success"] == True).mean()
        print_and_record(f"Contracts failed as declarer: Player={fail_rate_player:.2%}, Field={fail_rate_field:.2%}")
        print_and_record(f"Contracts made as declarer: Player={make_rate_player:.2%}, Field={make_rate_field:.2%}")
        path = plot_comparison(["Failed","Made"],[fail_rate_player,make_rate_player],[fail_rate_field,make_rate_field],"Overbid vs Underbid (As Declarer)","overbid_underbid.png","Rate")
        report_lines.append(f'<img src="{path}" width="600">')
    else:
        print_and_record("No declarer data available for comparison")

    # Contract Level Success
    print_and_record("\n--- Contract Level Success ---")
    success_by_level_player = df_boards.groupby("contract_level")["success"].mean()
    success_by_level_field = df_boards_field.groupby("contract_level")["success"].mean()
    cats = sorted(list(set(success_by_level_player.index) | set(success_by_level_field.index)))
    player_vals = [success_by_level_player.get(l, 0) for l in cats]
    field_vals = [success_by_level_field.get(l, 0) for l in cats]
    for i, lvl in enumerate(cats):
        print_and_record(f"Level {lvl}: Player={player_vals[i]:.2%}, Field={field_vals[i]:.2%}")
    path = plot_comparison(cats, player_vals, field_vals,"Success by Contract Level","contract_level_success.png","Success Rate")
    report_lines.append(f'<img src="{path}" width="600">')

    # Declarer vs Defender
    print_and_record("\n--- Declarer vs Defender ---")
    decl_player = df_boards[df_boards["is_defense"] == False]
    decl_field = df_boards_field[df_boards_field["is_defense"] == False]
    decl_success_player = decl_player["success"].mean() if not decl_player.empty else 0
    decl_success_field = decl_field["success"].mean() if not decl_field.empty else 0
    
    defe_player = df_boards[df_boards["is_defense"] == True]
    defe_field = df_boards_field[df_boards_field["is_defense"] == True]
    defe_success_player = (defe_player["over_under"] < 0).mean() if not defe_player.empty else 0
    defe_success_field = (defe_field["over_under"] < 0).mean() if not defe_field.empty else 0

    print_and_record(f"Contracts Made as Declarer: Player={decl_success_player:.2%}, Field={decl_success_field:.2%}")
    print_and_record(f"Contracts Set as Defender: Player={defe_success_player:.2%}, Field={defe_success_field:.2%}")
    path = plot_comparison(["Declarer (Made)","Defender (Set)"],[decl_success_player,defe_success_player],[decl_success_field,defe_success_field],"Declarer vs Defender","declarer_vs_defender.png","Success Rate")
    report_lines.append(f'<img src="{path}" width="600">')

    # Overtrick Efficiency
    print_and_record("\n--- Overtrick Efficiency ---")
    player_over_freq = (df_boards["over_under"] > 0).mean()
    field_over_freq = (df_boards_field["over_under"] > 0).mean()
    print_and_record(f"Overtrick frequency: Player={player_over_freq:.2%}, Field={field_over_freq:.2%}")
    path = plot_comparison(["Overtricks"],[player_over_freq],[field_over_freq],"Overtrick Efficiency","overtrick_efficiency.png","Frequency")
    report_lines.append(f'<img src="{path}" width="600">')

# === OUTPUT HTML ===
safe_name = re.sub(r"[^א-תA-Za-z0-9_\- ]", "", target_name)
out_fname = f"/home/ben/Desktop/Final_Project/player_stats_{safe_name.replace(' ', '_')}.html"
html_content = f"<html><head><meta charset='utf-8'><title>Stats for Player {target_name}</title></head><body><h1>Stats for Player {target_name}</h1>"
for line in report_lines:
    if line.strip().startswith("---"):
        html_content += f"<h2>{line.strip('- ').strip()}</h2>"
    elif line.strip().startswith("<img"):
        html_content += line
    else:
        html_content += f"<p>{line}</p>"
html_content += "</body></html>"

with open(out_fname, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Report written to {out_fname}")
