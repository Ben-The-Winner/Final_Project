import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os


target_name = "זיידנברג נצר יעקב"


folder_path = "bridge_data" 


restot_scores = []
low_hands = []
threshold = 60

for filename in os.listdir(folder_path):
    if filename.endswith(".xml"):
        filepath = os.path.join(folder_path, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        for pair in root.iter('pair'):
            names = pair.find('names').text
            if target_name in names:
                pair_id = pair.attrib['id']
                restot = float(pair.find('restot').text)
                restot_scores.append((filename, pair_id, restot))

                if restot < threshold:
                    low_hands.append((filename, pair_id, restot))


scores = [score for _, _, score in restot_scores]
if scores:
    avg = np.mean(scores)
    std_dev = np.std(scores)

    print(f"average score: {target_name}: {avg:.2f}")
    print(f"standard deviation: {std_dev:.2f}")
    print("\n hands with a score less than 60: ")
    for fname, pid, score in low_hands:
        print(f"file: {fname}, hand: {pid}, score: {score}")
else:
    print(f" couldn't find this player {target_name}")




results = []


for filename, pair_id, restot in low_hands:
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        continue 

    tree = ET.parse(filepath)
    root = tree.getroot()

    for pair in root.iter('pair'):
        if pair.attrib.get("id") == pair_id:
            names = pair.find("names").text
            ibfn1 = pair.find("ibfn1")
            ibfn2 = pair.find("ibfn2")

           
            ibfn1_tag = pair.find("ibfn1")
            ibfn2_tag = pair.find("ibfn2")

            
            ibfn1_val = ibfn1_tag.text.strip() if ibfn1_tag is not None else ""
            ibfn2_val = ibfn2_tag.text.strip() if ibfn2_tag is not None else ""

            
            if ibfn1_val == "13339":
                partner_tag = ibfn2_tag
            elif ibfn2_val == "13339":
                partner_tag = ibfn1_tag
            else:
                continue  


            partner_gender = partner_tag.attrib.get("G")
            partner_age = int(partner_tag.attrib.get("A"))
            partner_exp = int(partner_tag.attrib.get("E"))


            score = float(pair.find("restot").text)
            carry = float(pair.find("carry").text) if pair.find("carry").text else 0
            nmp = int(pair.find("nmp").text) if pair.find("nmp").text else 0

            results.append({
                "file": filename,
                "pair_id": pair_id,
                "score": score,
                "partner_gender": partner_gender,
                "partner_age": partner_age,
                "partner_experience": partner_exp,
                "carry": carry,
                "nmp": nmp,
                "partner_name": names.replace("זיידנברג נצר יעקב", "").replace("-", "").strip()
            })

df = pd.DataFrame(results)
print(df)

