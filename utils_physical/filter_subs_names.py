import pandas as pd
from pathlib import Path
import sys
# ✅ Fill this dictionary with your player name → code mapping
player_code_dict = {
        "Lior Refaelov": "AM_22", # rephaelov LEADER
        "L. Refaelov": "AM_22",
        "Dia Saba": "AM_21", #Sabia TERRORIST
        "D. Saba": "AM_21", #Sabia TERRORIST
        "Sean Goldberg": "CB_2", #goldberg NOTHING TO SAY
        "S. Goldberg": "CB_2", #goldberg NOTHING TO SAY
        "Abdoulaye Seck": "CB_3", #seck BLACK
        "A. Seck": "CB_3",
        "Oleksandr Syrota": "CB_5", #syrota WAR GUY
        "O. Syrota": "CB_5", #syrota WAR GUY
        "Dean David": "CF_34", #david OFFSIDE
        "D. David": "CF_34", #david OFFSIDE
        "Frantzdy Pierrot": "CF_37", #pierrot NOT TECHNICH
        "F. Pierrot": "CF_37",
        "Ali Mohamed": "DM_15", #mohammad LIKE HIM
        "A. Mohamed": "DM_15", #mohammad LIKE HIM
        "Ethane Azoulay": "DM_16", #azulay, VERY SHIT
        "E. Azoulay": "DM_16", #azulay, VERY SHIT
        "Kenny Saief": "LM_28", #Kenny Saief, GOOD ARABIC
        "K. Saief": "LM_28", #Kenny Saief, GOOD ARABIC
        "Iyad Khalaili": "RW_25", #Khalaili NOT GOOD LIKE HIS BROTHER
        "I. Khalaili": "RW_25",
        "Ilay Feingold": "RB_8", #Feingold AMERICAN
        "I. Feingold": "RB_8",
        "Dolev Haziza": "LW_29", #HAZIZA BAD
        "D. Haziza": "LW_29",
        "Gadi Kinda": "AM_20", #KINDA RIP
        "G. Kinda": "AM_20", #KINDA RIP
        "Mahmoud Jaber": "CM_18",#JABBER ARABIC
        "M. Jaber": "CM_18",#JABBER ARABIC
        "Liam Hermesh": "DM_17", #HERMESH WHO?
        "L. Hermesh": "DM_17", #HERMESH WHO?
        "Vital Nsimba": "LB_12", #NSIMBA BYE AND NEVER COME BACK
        "V. Nsimba": "LB_12",
        "Omer David Dahan": "CF_35", #DAHAN WHO?
        "O. D. Dahan": "CF_35",
        "Erik Shuranov": "CF_38", # SHURANOV SHIT
        "E. Shuranov": "CF_38",
        "Pedrao": "CB_6" , #PEDRAO BAD KNEES
        "Mathias Nahuel": "LW_30", #NAHUEL FLOP
        "M. Nahuel": "LW_30",
        "Maor Kandil": "RB_11", #KANDIL
        "M. Kandil": "RB_11",
        "Xander Severina": "RW_24", #SAVERINA
        "X. Severina": "RW_24", #SAVERINA
        "Ilay Hajaj": "RM_23", # HAGAG ALWAYS INJURED
        "I. Hajaj": "RM_23",
        "Roey Elimelech": "RB_10", #ELIMELECH
        "R. Elimelech": "RB_10", #ELIMELECH
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
        # CB_4 TOMER LANS
        #sharif kayuf the worst gk in the world
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
