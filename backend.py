import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from ddstable.ddstable import get_ddstable
import bisect

def board_to_pbn(board_elem):
    """
    Convert a <board> XML element into a PBN string that DDS can read.
    """
    dealer_map = {"North": "N", "East": "E", "South": "S", "West": "W"}
    dealer = dealer_map.get(board_elem.findtext("dealer"), "N")

    def fix_hand(hand):
        return hand if hand else "."

    north = fix_hand(board_elem.findtext("north"))
    east  = fix_hand(board_elem.findtext("east"))
    south = fix_hand(board_elem.findtext("south"))
    west  = fix_hand(board_elem.findtext("west"))

    return f"{dealer}:{north} {east} {south} {west}"

# === CONFIGURATION ===
folder_path = "/home/ben/Desktop/Final_Project/bridge_data"
plot_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(plot_dir, exist_ok=True)

os.makedirs(plot_dir, exist_ok=True)

# === NORMALIZE HEBREW ===
def normalize_hebrew(s):
    if not s:
        return ""
    s = re.sub(r"[^א-ת\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# === PLAYER ANALYSIS WRAPPER ===
def run_analysis_for_player(player_identifier):
    """
    Run the full bridge performance analysis for a given player name or number.
    Returns the HTML report file path.
    """

    # --- Determine player name or IBFN ---
    if isinstance(player_identifier, int) or (isinstance(player_identifier, str) and player_identifier.isdigit()):
        target_ibfn = str(player_identifier)
        target_name = None
    else:
        target_ibfn = None
        target_name = player_identifier

    target_name_norm = normalize_hebrew(target_name) if target_name else None

# === MAIN ANALYSIS LOGIC ===
  
    # === UTILITIES ===
    def parse_contract(contract_str):
        if not contract_str:
            return None
        cs = contract_str.strip()
        m = re.match(r"^(\d)([SHDCN])(X{0,2})([NESW])([=+-]\d+|=)?$", cs)
        if not m:
            return None
        level = int(m.group(1))
        strain = m.group(2)
        doubled = m.group(3)  # X or XX or empty
        declarer = m.group(4)
        result_str = m.group(5) or "="
        tricks_bid = level + 6
        tricks_made = tricks_bid
        if result_str.startswith("+"):
            tricks_made += int(result_str[1:])
        elif result_str.startswith("-"):
            tricks_made -= int(result_str[1:])
        return {
            "contract_level": level,
            "strain": strain,
            "doubled": doubled,
            "declarer": declarer,
            "tricks_bid": tricks_bid,
            "tricks_made": tricks_made,
            "over_under": tricks_made - tricks_bid,
            "success": tricks_made >= tricks_bid,
            "raw": contract_str
        }

    def calculate_score(contract_level, strain, doubled, declarer_vulnerable, tricks_made, tricks_bid):
        """Calculate bridge score for a contract"""
        if tricks_made < tricks_bid:
            # Contract failed - calculate penalty
            undertricks = tricks_bid - tricks_made
            if not doubled:
                if declarer_vulnerable:
                    penalty = undertricks * 100
                else:
                    penalty = undertricks * 50
            elif doubled == "X":
                if declarer_vulnerable:
                    penalty = 200 + (undertricks - 1) * 300
                else:
                    penalty = 100 + (undertricks - 1) * 200
            else:  # XX
                penalty = penalty * 2 if doubled == "X" else 0
            return -penalty
        
        # Contract made
        overtricks = tricks_made - tricks_bid
        
        # Base score
        if strain in ['C', 'D']:
            base_per_trick = 20
        elif strain in ['H', 'S']:
            base_per_trick = 30
        else:  # NT
            base_per_trick = 30
            base_score = 40 + (contract_level - 1) * 30  # First trick is 40
        
        if strain != 'N':
            base_score = contract_level * base_per_trick
        
        # Apply doubling
        if doubled == "X":
            base_score *= 2
        elif doubled == "XX":
            base_score *= 4
        
        # Game/part-game bonus
        if base_score >= 100:
            game_bonus = 500 if declarer_vulnerable else 300
        else:
            game_bonus = 50
        
        # Slam bonus
        slam_bonus = 0
        if contract_level == 6:
            slam_bonus = 750 if declarer_vulnerable else 500
        elif contract_level == 7:
            slam_bonus = 1500 if declarer_vulnerable else 1000
        
        # Overtricks
        if not doubled:
            if strain in ['C', 'D']:
                overtrick_value = overtricks * 20
            else:
                overtrick_value = overtricks * 30
        elif doubled == "X":
            overtrick_value = overtricks * (200 if declarer_vulnerable else 100)
        else:  # XX
            overtrick_value = overtricks * (400 if declarer_vulnerable else 200)
        
        # Double/redouble bonus
        double_bonus = 50 if doubled == "X" else 100 if doubled == "XX" else 0
        
        total = base_score + game_bonus + slam_bonus + overtrick_value + double_bonus
        return total

    def get_declarer_side(declarer):
        if not declarer:
            return None
        return "NS" if declarer in ("N", "S") else "EW"

    def rotate_pbn_to_north(pbn_text):
        """Ensure PBN always starts with North"""
        if not pbn_text:
            return pbn_text
        
        if pbn_text.startswith("N:"):
            return pbn_text
        
        parts = pbn_text.split()
        if len(parts) != 5:
            return pbn_text
        
        seat_order = parts[0][0]
        hands = parts[1:]
        
        seat_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        current_idx = seat_map.get(seat_order, 0)
        
        rotated_hands = hands[-current_idx:] + hands[:-current_idx]
        return f"N:{rotated_hands[0]} E:{rotated_hands[1]} S:{rotated_hands[2]} W:{rotated_hands[3]}"

    def get_dds_tricks(deal_pbn, declarer, strain, dds_cache):
        """Get double dummy tricks with caching"""
        if not deal_pbn or not declarer or not strain:
            return None
        
        # Check cache first
        cache_key = (deal_pbn, declarer, strain)
        if cache_key in dds_cache:
            return dds_cache[cache_key]
        
        try:
            pbn = rotate_pbn_to_north(deal_pbn)
            
            if 'N:' in pbn and len(re.findall(r'\.', pbn)) == 12:
                dds_result = get_ddstable(pbn.encode("utf-8"))
                strain_key = "NT" if strain == "N" else strain.upper()
                
                if declarer in dds_result and strain_key in dds_result[declarer]:
                    tricks = dds_result[declarer][strain_key]
                    dds_cache[cache_key] = tricks
                    return tricks
                else:
                    dds_cache[cache_key] = None
                    return None
            else:
                dds_cache[cache_key] = None
                return None
        except Exception as e:
            dds_cache[cache_key] = None
            return None

    def get_dds_all_tricks(deal_pbn, dds_cache):
        """Get full DDS table for all declarers and strains"""
        if not deal_pbn:
            return None
        
        # Check if we already computed full table for this deal
        full_cache_key = (deal_pbn, "FULL")
        if full_cache_key in dds_cache:
            return dds_cache[full_cache_key]
        
        try:
            pbn = rotate_pbn_to_north(deal_pbn)
            
            if 'N:' in pbn and len(re.findall(r'\.', pbn)) == 12:
                dds_result = get_ddstable(pbn.encode("utf-8"))
                dds_cache[full_cache_key] = dds_result
                return dds_result
            else:
                dds_cache[full_cache_key] = None
                return None
        except Exception as e:
            dds_cache[full_cache_key] = None
            return None

    def get_optimal_level(deal_pbn, side, dds_cache):
        """Get optimal contract level for the side"""
        if not deal_pbn or not side:
            return None
        
        try:
            dds_result = get_dds_all_tricks(deal_pbn, dds_cache)
            if not dds_result:
                return None
            
            declarers = ['N', 'S'] if side == "NS" else ['E', 'W']
            strains = ['NT', 'S', 'H', 'D', 'C']
            
            max_tricks = 0
            for d in declarers:
                for s in strains:
                    if d in dds_result and s in dds_result[d]:
                        max_tricks = max(max_tricks, dds_result[d][s])
            
            optimal_level = max(0, max_tricks - 6)
            return optimal_level
        except Exception as e:
            return None

    # === DATA EXTRACTION ===
    all_results = []
    board_records = []
    board_records_field = []
    pbn_count = 0
    dds_success_count = 0

    target_name_norm = normalize_hebrew(target_name)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(folder_path, filename)
        
        # Create a DDS cache for this file
        dds_cache = {}
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue

        found_player = False
        target_pair_id = None
        target_score = np.nan

        # Loop through all pairs and find matching player by name or IBFN
        for pair in root.iter("pair"):
            names = pair.findtext("names") or ""
            ibfn1 = pair.findtext("ibfn1") or ""
            ibfn2 = pair.findtext("ibfn2") or ""
            rank = pair.findtext("rank")
            score = pair.findtext("restot")

            if target_name_norm and target_name_norm in normalize_hebrew(names):
                found_player = True
            elif target_ibfn and (target_ibfn in ibfn1 or target_ibfn in ibfn2):
                found_player = True

            if found_player:
                target_pair_id = pair.attrib.get("id")
                target_score = float(score) if score else np.nan
                all_results.append({
                    "file": filename,
                    "pair_id": target_pair_id,
                    "score": target_score,
                    "ibfn1": ibfn1,
                    "ibfn2": ibfn2,
                    "names": names
                })
                print(f"Found player in {filename}: {names} (pair_id={target_pair_id}, score={target_score})")
                break

        # Pre-calculate DDS for all unique hands in this file
        unique_hands = set()
        for board in root.iter('board'):
            deal_pbn = board_to_pbn(board)
            if deal_pbn:
                unique_hands.add(deal_pbn)
        
        print(f"Pre-calculating DDS for {len(unique_hands)} unique hands in {filename}...")
        for hand_pbn in unique_hands:
            dds_all = get_dds_all_tricks(hand_pbn, dds_cache)
            if dds_all:
                dds_success_count += 1
        
        # collect board-level field data
        b_id = 0
        for board in root.iter('board'):
            b_id += 1
            deal_pbn = board_to_pbn(board)
            
            if deal_pbn:
                pbn_count += 1
            
            for data in board.findall('data'):
                contract_str_all = data.attrib.get("C") or ""
                parsed_all = parse_contract(contract_str_all)
                declarer = parsed_all["declarer"] if parsed_all else None
                side = "NS" if data.attrib.get('N') or data.attrib.get('Nss') else "EW" if data.attrib.get('E') or data.attrib.get('Ews') else None
                if not side:
                    continue
                declarer_side = get_declarer_side(declarer)
                is_defense = side != declarer_side if declarer_side else None
                
                field_record = {
                    "file": filename,
                    "board_id": str(b_id),
                    "side": side,
                    "contract_raw": contract_str_all,
                    "declarer_side": declarer_side,
                    "is_defense": is_defense,
                    "deal_pbn": deal_pbn,
                    "ns_vulnerable": data.attrib.get("Nv", "0") == "1",
                    "ew_vulnerable": data.attrib.get("Ev", "0") == "1",
                }
                
                if parsed_all:
                    field_record.update(parsed_all)
                
                # Use cache for DDS lookups
                if parsed_all and declarer and deal_pbn:
                    dds_tricks = get_dds_tricks(deal_pbn, declarer, parsed_all["strain"], dds_cache)
                    if dds_tricks is not None:
                        field_record["dds_tricks"] = dds_tricks
                
                board_records_field.append(field_record)

        # boards of this pair
        if target_pair_id:
            b_id = 0
            for board in root.iter('board'):
                b_id += 1
                deal_pbn = board_to_pbn(board)
                
                for data in board.findall('data'):
                    if data.attrib.get('N') == target_pair_id or data.attrib.get('E') == target_pair_id:
                        side = "NS" if data.attrib.get('N') == target_pair_id else "EW"
                        player_pct = float(data.attrib.get("Nss" if side == "NS" else "Ews", 0))
                        contract_str = data.attrib.get("C") or ""
                        parsed = parse_contract(contract_str)
                        declarer = parsed["declarer"] if parsed else None
                        declarer_side = get_declarer_side(declarer)
                        is_defense = side != declarer_side if declarer_side else None

                        player_record = {
                            "file": filename,
                            "board_id": str(b_id),
                            "pair_id": target_pair_id,
                            "side": side,
                            "player_pct": player_pct,
                            "contract_raw": contract_str,
                            "declarer_side": declarer_side,
                            "is_defense": is_defense,
                            "deal_pbn": deal_pbn,
                            "ns_vulnerable": data.attrib.get("Nv", "0") == "1",
                            "ew_vulnerable": data.attrib.get("Ev", "0") == "1",
                        }
                        
                        if parsed:
                            player_record.update(parsed)
                        
                        # Use cache for DDS lookups
                        if parsed and declarer and deal_pbn:
                            dds_tricks = get_dds_tricks(deal_pbn, declarer, parsed["strain"], dds_cache)
                            if dds_tricks is not None:
                                player_record["dds_tricks"] = dds_tricks
                                dds_success_count += 1
                            
                            if not is_defense:
                                optimal_level = get_optimal_level(deal_pbn, side, dds_cache)
                                if optimal_level is not None:
                                    player_record["optimal_level"] = optimal_level
                        
                        board_records.append(player_record)

    print(f"Total PBNs parsed: {pbn_count}")
    print(f"Total DDS successes: {dds_success_count}")

    # === DATAFRAMES ===
    df_all = pd.DataFrame(all_results)
    df_boards = pd.DataFrame(board_records)
    df_boards_field = pd.DataFrame(board_records_field)

    print(f"Found {len(df_all)} total results for player")
    print(f"Found {len(df_boards)} board records for player")
    print(f"Found {len(df_boards_field)} field board records")

    # Debug DDS data
    dds_count_player = 0
    dds_count_field = 0
    if 'dds_tricks' in df_boards.columns:
        dds_count_player = df_boards['dds_tricks'].notna().sum()
    if 'dds_tricks' in df_boards_field.columns:
        dds_count_field = df_boards_field['dds_tricks'].notna().sum()

    print(f"Player boards with DDS data: {dds_count_player}/{len(df_boards)}")
    print(f"Field boards with DDS data: {dds_count_field}/{len(df_boards_field)}")

    # === REPORT ===
    report_lines = []
    def print_and_record(s=""):
        print(s)
        report_lines.append(s)

    def plot_comparison(categories, player_vals, field_vals, dds_vals, title, filename, ylabel="", show_dds=True):
        x = np.arange(len(categories))
        width = 0.25
        plt.figure(figsize=(10, 6))
        
        # Replace NaN with 0 for display
        player_vals = [v if not np.isnan(v) else 0 for v in player_vals]
        field_vals = [v if not np.isnan(v) else 0 for v in field_vals]
        
        if show_dds and len(dds_vals) == len(categories):
            dds_vals = [v if not np.isnan(v) else 0 for v in dds_vals]
        else:
            dds_vals = []  # prevent plotting DDS when not needed

        # Plot bars
        bars1 = plt.bar(x - width, player_vals, width, label="Player", color='#2E86AB', alpha=0.8)
        bars2 = plt.bar(x, field_vals, width, label="Field", color='#A23B72', alpha=0.8)
        
        bars3 = []
        if show_dds and dds_vals:
            bars3 = plt.bar(x + width, dds_vals, width, label="DDS Optimal", color='#F18F01', alpha=0.8)
        
        plt.xticks(x, categories, rotation=45)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Annotate values on bars
        for bars, vals in [(bars1, player_vals), (bars2, field_vals), (bars3, dds_vals)]:
            for bar, val in zip(bars, vals):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        path = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        
        # FIXED: Convert to relative web path for HTML
        relative_path = os.path.relpath(path, os.path.join(os.path.dirname(__file__), "static"))
        return f'/static/{relative_path}'

    if df_all.empty:
        print(f"Couldn't find {target_name}")
    else:
        player_declarer = df_boards[df_boards["is_defense"] == False].copy()
        player_defender = df_boards[df_boards["is_defense"] == True].copy()
        field_declarer = df_boards_field[df_boards_field["is_defense"] == False].copy()
        field_defender = df_boards_field[df_boards_field["is_defense"] == True].copy()
        
        print(f"Player declarer records: {len(player_declarer)}")
        print(f"Player defender records: {len(player_defender)}")
        print(f"Field declarer records: {len(field_declarer)}")
        print(f"Field defender records: {len(field_defender)}")

        # === AGGRESSION ===
        print_and_record("\n--- Aggression ---")
        avg_level_player = player_declarer["contract_level"].mean() if not player_declarer.empty and 'contract_level' in player_declarer.columns else np.nan
        avg_level_field = field_declarer["contract_level"].mean() if not field_declarer.empty and 'contract_level' in field_declarer.columns else np.nan
        avg_level_dds = player_declarer["optimal_level"].mean() if not player_declarer.empty and 'optimal_level' in player_declarer.columns else np.nan
        print_and_record(f"Average contract level: Player={avg_level_player:.2f}, Field={avg_level_field:.2f}, DDS={avg_level_dds:.2f}")
        path = plot_comparison(["Contract Level"], [avg_level_player], [avg_level_field], [avg_level_dds], "Aggression", "aggression.png", "Avg Level")
        relative_path = os.path.relpath(path, os.path.join(os.path.dirname(__file__), "static"))
        report_lines.append(f'<img src="/static/{relative_path}" width="600">')


        # === DECLARER DOUBLE DUMMY DIFFERENCE ===
        print_and_record("\n--- Declarer Double Dummy Difference ---")
        player_declarer_dd = player_declarer.dropna(subset=['dds_tricks', 'tricks_made'])
        field_declarer_dd = field_declarer.dropna(subset=['dds_tricks', 'tricks_made'])
        
        if not player_declarer_dd.empty:
            player_declarer_dd['dd_diff'] = player_declarer_dd['tricks_made'] - player_declarer_dd['dds_tricks']
            avg_dd_diff_player = player_declarer_dd['dd_diff'].mean()
            print_and_record(f"Player declarer DD difference (actual - optimal): {avg_dd_diff_player:.2f} tricks")
            print_and_record(f"Based on {len(player_declarer_dd)} boards with DDS data")
        else:
            avg_dd_diff_player = np.nan
            print_and_record("No player declarer DDS data available")
        
        if not field_declarer_dd.empty:
            field_declarer_dd['dd_diff'] = field_declarer_dd['tricks_made'] - field_declarer_dd['dds_tricks']
            avg_dd_diff_field = field_declarer_dd['dd_diff'].mean()
            print_and_record(f"Field declarer DD difference (actual - optimal): {avg_dd_diff_field:.2f} tricks")
        else:
            avg_dd_diff_field = np.nan
            print_and_record("No field declarer DDS data available")
        
        avg_dd_diff_dds = 0.0
        
        path = plot_comparison(["Declarer DD Diff"], [avg_dd_diff_player], [avg_dd_diff_field], [avg_dd_diff_dds], 
                            "Declarer Double Dummy Difference", "declarer_dd_diff.png", "Avg Tricks (Actual - Optimal)")
        report_lines.append(f'<img src="{path}" width="600">')

        # === DEFENDER DOUBLE DUMMY DIFFERENCE ===
        print_and_record("\n--- Defender Double Dummy Difference ---")
        player_defender_dd = player_defender.dropna(subset=['dds_tricks', 'tricks_made'])
        field_defender_dd = field_defender.dropna(subset=['dds_tricks', 'tricks_made'])
        
        if not player_defender_dd.empty:
            player_defender_dd['dd_diff'] = player_defender_dd['tricks_made'] - player_defender_dd['dds_tricks']
            avg_dd_diff_def_player = player_defender_dd['dd_diff'].mean()
            print_and_record(f"Player defender DD difference (tricks given away): {avg_dd_diff_def_player:.2f} tricks")
            print_and_record(f"Based on {len(player_defender_dd)} boards with DDS data")
        else:
            avg_dd_diff_def_player = np.nan
            print_and_record("No player defender DDS data available")
        
        if not field_defender_dd.empty:
            field_defender_dd['dd_diff'] = field_defender_dd['tricks_made'] - field_defender_dd['dds_tricks']
            avg_dd_diff_def_field = field_defender_dd['dd_diff'].mean()
            print_and_record(f"Field defender DD difference (tricks given away): {avg_dd_diff_def_field:.2f} tricks")
        else:
            avg_dd_diff_def_field = np.nan
            print_and_record("No field defender DDS data available")
        
        avg_dd_diff_def_dds = 0.0
        
        path = plot_comparison(["Defender DD Diff"], [avg_dd_diff_def_player], [avg_dd_diff_def_field], [avg_dd_diff_def_dds], 
                            "Defender Double Dummy Difference", "defender_dd_diff.png", "Avg Tricks Given Away")
        report_lines.append(f'<img src="{path}" width="600">')

        # === TRICK LOSING PLAYS - OPENING LEAD ===
        print_and_record("\n--- Trick Losing Plays - Opening Lead ---")
        player_defender_lead = player_defender.dropna(subset=['dds_tricks', 'tricks_made', 'declarer', 'side'])
        field_defender_lead = field_defender.dropna(subset=['dds_tricks', 'tricks_made', 'declarer', 'side'])
        
        lead_map = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        
        tlp_lead_count_player = 0
        tlp_lead_total_player = 0
        
        for idx, row in player_defender_lead.iterrows():
            opening_leader = lead_map.get(row['declarer'])
            if opening_leader:
                player_side = row['side']
                is_opening_leader = (player_side == "NS" and opening_leader in ['N', 'S']) or \
                                (player_side == "EW" and opening_leader in ['E', 'W'])
                
                if is_opening_leader:
                    tlp_lead_total_player += 1
                    if row['tricks_made'] > row['dds_tricks']:
                        tlp_lead_count_player += 1
        
        tlp_lead_count_field = 0
        tlp_lead_total_field = 0
        
        for idx, row in field_defender_lead.iterrows():
            opening_leader = lead_map.get(row['declarer'])
            if opening_leader:
                player_side = row['side']
                is_opening_leader = (player_side == "NS" and opening_leader in ['N', 'S']) or \
                                (player_side == "EW" and opening_leader in ['E', 'W'])
                
                if is_opening_leader:
                    tlp_lead_total_field += 1
                    if row['tricks_made'] > row['dds_tricks']:
                        tlp_lead_count_field += 1
        
        tlp_lead_rate_player = tlp_lead_count_player / tlp_lead_total_player if tlp_lead_total_player > 0 else np.nan
        tlp_lead_rate_field = tlp_lead_count_field / tlp_lead_total_field if tlp_lead_total_field > 0 else np.nan
        tlp_lead_rate_dds = 0.0
        
        print_and_record(f"Player TLP on opening lead: {tlp_lead_rate_player:.2%} ({tlp_lead_count_player}/{tlp_lead_total_player} boards)")
        print_and_record(f"Field TLP on opening lead: {tlp_lead_rate_field:.2%} ({tlp_lead_count_field}/{tlp_lead_total_field} boards)")
        
        path = plot_comparison(["Opening Lead TLP"], [tlp_lead_rate_player], [tlp_lead_rate_field], [tlp_lead_rate_dds], 
                            "Trick Losing Plays - Opening Lead", "tlp_opening_lead.png", "Frequency")
        report_lines.append(f'<img src="{path}" width="600">')

        # === TRICK LOSING PLAYS - AS DEFENDER (excluding opening lead) ===
        print_and_record("\n--- Trick Losing Plays - As Defender (Excluding Opening Lead) ---")
        
        tlp_def_count_player = 0
        tlp_def_total_player = len(player_defender_dd)
        
        if not player_defender_dd.empty:
            tlp_def_count_player = (player_defender_dd['tricks_made'] > player_defender_dd['dds_tricks']).sum()
            tlp_def_rate_player = tlp_def_count_player / tlp_def_total_player if tlp_def_total_player > 0 else np.nan
            print_and_record(f"Player TLP as defender: {tlp_def_rate_player:.2%} ({tlp_def_count_player}/{tlp_def_total_player} boards)")
        else:
            tlp_def_rate_player = np.nan
            print_and_record("No player defender TLP data available")
        
        tlp_def_count_field = 0
        tlp_def_total_field = len(field_defender_dd)
        
        if not field_defender_dd.empty:
            tlp_def_count_field = (field_defender_dd['tricks_made'] > field_defender_dd['dds_tricks']).sum()
            tlp_def_rate_field = tlp_def_count_field / tlp_def_total_field if tlp_def_total_field > 0 else np.nan
            print_and_record(f"Field TLP as defender: {tlp_def_rate_field:.2%} ({tlp_def_count_field}/{tlp_def_total_field} boards)")
        else:
            tlp_def_rate_field = np.nan
            print_and_record("No field defender TLP data available")
        
        tlp_def_rate_dds = 0.0
        
        path = plot_comparison(["Defender TLP"], [tlp_def_rate_player], [tlp_def_rate_field], [tlp_def_rate_dds], 
                            "Trick Losing Plays - As Defender", "tlp_defender.png", "Frequency")
        report_lines.append(f'<img src="{path}" width="600">')

        # === RESULTS WITHOUT PLAY ===
        print_and_record("\n--- Results Without Play (Contract-Only Analysis, Cross-IMPs) ---")
        def score_to_imp_continuous(diff):
            """Convert score difference to fractional IMPs using linear interpolation."""
            imp_table = [
                (0, 0), (20, 1), (50, 2), (90, 3), (130, 4), (170, 5),
                (220, 6), (270, 7), (320, 8), (370, 9), (430, 10),
                (500, 11), (600, 12), (750, 13), (900, 14), (1100, 15),
                (1300, 16), (1500, 17), (1750, 18), (2000, 19), (2250, 20),
                (2500, 21), (3000, 22), (3500, 23), (4000, 24)
            ]

            sign = 1 if diff >= 0 else -1
            abs_diff = abs(diff)
            diffs = [d for d, _ in imp_table]
            imps = [i for _, i in imp_table]

            if abs_diff >= diffs[-1]:
                return sign * imps[-1]
            if abs_diff <= diffs[0]:
                return 0.0

            idx = bisect.bisect_left(diffs, abs_diff)
            d1, i1 = diffs[idx-1], imps[idx-1]
            d2, i2 = diffs[idx], imps[idx]
            # Linear interpolation
            imp_value = i1 + (abs_diff - d1) * (i2 - i1) / (d2 - d1)
            return sign * imp_value


        player_boards_with_contract = df_boards.dropna(subset=[
            'contract_level', 'strain', 'doubled', 'declarer',
            'tricks_made', 'tricks_bid', 'side'
        ])

        cross_imps = []

        for idx, player_row in player_boards_with_contract.iterrows():
            board_id = player_row['board_id']
            file_name = player_row['file']
            player_side = player_row['side']

            # All field results for the same board
            board_field_results = df_boards_field[
                (df_boards_field['board_id'] == board_id) &
                (df_boards_field['file'] == file_name)
            ].copy()
            if board_field_results.empty:
                continue

            # Compute “no-play” scores for each field result
            board_field_results['declarer_vul'] = board_field_results.apply(
                lambda r: r['ns_vulnerable'] if r['declarer_side'] == 'NS' else r['ew_vulnerable'], axis=1
            )
            board_field_results['score_no_play'] = board_field_results.apply(
                lambda r: calculate_score(r['contract_level'], r['strain'], r['doubled'],
                                        r['declarer_vul'], r['tricks_bid'], r['tricks_bid']),
                axis=1
            )

            # Player’s own “no-play” score
            player_vul = player_row['ns_vulnerable'] if player_side == 'NS' else player_row['ew_vulnerable']
            player_score_no_play = calculate_score(
                player_row['contract_level'], player_row['strain'], player_row['doubled'],
                player_vul, player_row['tricks_bid'], player_row['tricks_bid']
            )

            if player_row['is_defense']:
                player_score_no_play = -player_score_no_play

            scores = board_field_results['score_no_play'].values
            if len(scores) < 2:
                continue

            # === Cross-IMPs: compare player's score vs all others ===
            imps_sum = 0
            for s in scores:
                if np.isnan(s) or s == player_score_no_play:
                    continue
                imps_sum += score_to_imp_continuous(player_score_no_play - s)

            cross_imp = imps_sum / (len(scores) - 1)
            cross_imps.append(cross_imp)

        # === Aggregate and Report ===
        if cross_imps:
            avg_cross_imp = np.mean(cross_imps)
            print_and_record(f"Average Cross-IMPs (Results Without Play): {avg_cross_imp:.2f}")
            print_and_record(f"Based on {len(cross_imps)} comparable boards")

            path = plot_comparison(["Result Without Play"], [avg_cross_imp], [0], [],#no dds optimal chart
                                "Results Without Play (Contract Quality, Cross-IMPs)",
                                "results_without_play.png", "Avg IMPs", show_dds = False)
            report_lines.append(f'<img src="{path}" width="600">')
        else:
            print_and_record("Insufficient data for Results Without Play (Cross-IMPs) analysis")


        # === MAKING POST-LEAD MAKEABLE CONTRACTS ===
        print_and_record("\n--- Making Post-Lead Makeable Contracts ---")
        
        player_declarer_postlead = player_declarer.dropna(subset=['dds_tricks', 'tricks_made', 'tricks_bid'])
        field_declarer_postlead = field_declarer.dropna(subset=['dds_tricks', 'tricks_made', 'tricks_bid'])
        
        player_makeable = player_declarer_postlead[player_declarer_postlead['dds_tricks'] >= player_declarer_postlead['tricks_bid']]
        field_makeable = field_declarer_postlead[field_declarer_postlead['dds_tricks'] >= field_declarer_postlead['tricks_bid']]
        
        if not player_makeable.empty:
            player_made_makeable = (player_makeable['tricks_made'] >= player_makeable['tricks_bid']).sum()
            player_total_makeable = len(player_makeable)
            player_rate = player_made_makeable / player_total_makeable if player_total_makeable > 0 else np.nan
            
            print_and_record(f"Player making post-lead makeable contracts: {player_rate:.2%} ({player_made_makeable}/{player_total_makeable})")
        else:
            player_rate = np.nan
            print_and_record("No player data for post-lead makeable contracts")
        
        if not field_makeable.empty:
            field_made_makeable = (field_makeable['tricks_made'] >= field_makeable['tricks_bid']).sum()
            field_total_makeable = len(field_makeable)
            field_rate = field_made_makeable / field_total_makeable if field_total_makeable > 0 else np.nan
            
            print_and_record(f"Field making post-lead makeable contracts: {field_rate:.2%} ({field_made_makeable}/{field_total_makeable})")
        else:
            field_rate = np.nan
            print_and_record("No field data for post-lead makeable contracts")
        
        dds_rate = 1.0
        
        if not np.isnan(player_rate):
            path = plot_comparison(["Post-Lead Makeable"], [player_rate], [field_rate], [dds_rate], 
                                "Making Post-Lead Makeable Contracts", "postlead_makeable.png", "Success Rate")
            report_lines.append(f'<img src="{path}" width="600">')

    # === OUTPUT HTML ===
    safe_name = re.sub(r"[^א-תA-Za-z0-9_\- ]", "", target_name or str(target_ibfn))
    out_fname = f"/home/ben/Desktop/Final_Project/player_stats_{safe_name.replace(' ', '_')}.html"
    html_content = f"""<!DOCTYPE html>
    <html><head><meta charset='utf-8'><title>Stats for Player {target_name}</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
    h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 30px; }}
    p {{ line-height: 1.6; margin: 10px 0; }}
    img {{ display: block; margin: 20px auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .debug {{ background-color: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 0.9em; }}
    form {{ text-align: center; margin-bottom: 25px; }}
    input, button {{ padding: 6px 10px; font-size: 1em; }}
    #status {{ color: #555; font-style: italic; margin-top: 10px; }}
    </style>
    </head><body>

    <!-- ===== Player Input Form ===== -->
    <form id="playerForm">
    <label for="playerInput"><b>Enter player name or number:</b></label><br>
    <input type="text" id="playerInput" name="playerInput" placeholder="" style="width:250px;">
    <button type="button" onclick="startAnalysis()">Run Analysis</button>
    </form>

    <div id="status"><i>Waiting for input...</i></div>

    <script>
    function startAnalysis() {{
    var player = document.getElementById("playerInput").value;
    document.getElementById("status").innerHTML =
        "<b>Running analysis for " + player + "...</b><br><i>This is gonna take a while. Please wait.</i>";
    }}
    </script>

    <!-- ===== Report Header ===== -->
    <h1>Bridge Performance Analysis for {target_name or f'Player #{target_ibfn}'}</h1>
    """


    html_content += f"""
    <div class="debug">
        <strong>Debug Info:</strong> {len(df_boards)} player boards, {dds_count_player} with DDS data<br>
        Field: {len(df_boards_field)} boards, {dds_count_field} with DDS data<br>
        PBNs parsed: {pbn_count}, DDS successes: {dds_success_count}
    </div>
    """

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

    return out_fname

if __name__ == "__main__":
    player_identifier = input("Enter player name or number: ").strip()
    run_analysis_for_player(player_identifier)

