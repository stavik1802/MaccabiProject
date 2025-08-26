import pandas as pd
from pathlib import Path
import sys
# ✅ Fill this dictionary with your player name → code mapping
player_code_dict = {
        "Lior Refaelov": "AM_22", 
        "L. Refaelov": "AM_22",
        "Dia Saba": "AM_21", 
        "D. Saba": "AM_21", 
        "Sean Goldberg": "CB_2", 
        "S. Goldberg": "CB_2", 
        "Abdoulaye Seck": "CB_3", 
        "A. Seck": "CB_3",
        "Oleksandr Syrota": "CB_5", 
        "O. Syrota": "CB_5", 
        "Dean David": "CF_34", 
        "D. David": "CF_34",
        "Frantzdy Pierrot": "CF_37", 
        "F. Pierrot": "CF_37",
        "Ali Mohamed": "DM_15", 
        "A. Mohamed": "DM_15", 
        "Ethane Azoulay": "DM_16", 
        "E. Azoulay": "DM_16", 
        "Kenny Saief": "LM_28", 
        "K. Saief": "LM_28", 
        "Iyad Khalaili": "RW_25", 
        "I. Khalaili": "RW_25",
        "Ilay Feingold": "RB_8", 
        "I. Feingold": "RB_8",
        "Dolev Haziza": "LW_29", 
        "D. Haziza": "LW_29",
        "Gadi Kinda": "AM_20", 
        "G. Kinda": "AM_20", 
        "Mahmoud Jaber": "CM_18",
        "M. Jaber": "CM_18",
        "Liam Hermesh": "DM_17", 
        "L. Hermesh": "DM_17", 
        "Vital Nsimba": "LB_12", 
        "V. Nsimba": "LB_12",
        "Omer David Dahan": "CF_35", 
        "O. D. Dahan": "CF_35",
        "Erik Shuranov": "CF_38", 
        "E. Shuranov": "CF_38",
        "Pedrao": "CB_6" , 
        "Mathias Nahuel": "LW_30", 
        "M. Nahuel": "LW_30",
        "Maor Kandil": "RB_11", 
        "M. Kandil": "RB_11",
        "Xander Severina": "RW_24", 
        "X. Severina": "RW_24", 
        "Ilay Hajaj": "RM_23", 
        "I. Hajaj": "RM_23",
        "Roey Elimelech": "RB_10", 
        "R. Elimelech": "RB_10", 
        "D. Sundgren": "RB_9",
        "Daniel Sundgren": "RB_9",
        "Guy Melamed": "CF_36",
        "G. Melamed": "CF_36",
        "Ricardinho": "CF_33",
        "Rami Gershon": "CB_7",
        "R. Gershon": "CB_7",
        "Tomer Lannes": "CB_4",
        "T. Lannes": "CB_4",
        "A. Khalaili": "RW_46",
        "Anan Khalaili": "RW_46",
        "Hamza Shibli": "LW_31",
        "H. Shibli": "LW_31",
        "Show": "DM_42",
        "Suf Podgoreanu": "UNK_39",
        "S. Podgoreanu": "UNK_39",
        "Lorenco Šimić": "CB_41",
        "L. Šimić": "CB_41",
        "Tjaronn Chery": "AM_45",
        "T. Chery": "AM_45",
        "Pierre Cornud": "LB_13",
        "P. Cornud": "LB_13",
        "Goni Naor": "DM_14",
        "G. Naor": "DM_14",
        "Tomer Hemed": "CF_48",
        "T. Hemed": "CF_48",
        "Lior Kasa": "DM_43",
        "L. Kasa": "DM_43",
        "Daniil Lesovoy": "LW_47",
        "D. Lesovoy": "LW_47",
        "Ziv Ben Shimol" :"AM_44",
        "Z. B. Shimol": "AM_44"

}

def replace_players_in_folder(folder_path):
    folder = Path(folder_path)
    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)

        # Replace player names if columns exist
        if "In Player" in df.columns:
            df["In Player"] = df["In Player"].map(player_code_dict).fillna(df["In Player"])
        if "Out Player" in df.columns:
            df["Out Player"] = df["Out Player"].map(player_code_dict).fillna(df["Out Player"])

        df.to_csv(file, index=False)  # Overwrite original file
        print(f"Updated: {file.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replace_players.py /path/to/folder")
    else:
        replace_players_in_folder(sys.argv[1])
