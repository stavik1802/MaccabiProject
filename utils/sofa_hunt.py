# open website with selenium
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

#this script is used to extract the lineup and substitutes from a game
#you can change the url to the game you want to scrape
#the script will save the lineup and substitutes in a csv file
#the csv file will be saved in the same folder as the script
#the csv file will be named with the date of the game and the team names
#the csv file will have the following columns: Team, Player, Shirt Number
#the csv file will have the following columns: Team, In Player, Minute, Out Player

# URL of the game to scrape
url_str = "https://www.sofascore.com/football/match/maccabi-netanya-maccabi-haifa/XdcsVhc#id:12672675"

def extract_players_info(module_sub_string):
    """
    Extracts players' names and shirt numbers from the specified module sub-string.
    """
    team_players, team_shirts = [], []
    for team_level in driver.find_elements("css selector", module_sub_string):
        players_info = team_level.text.split("\n")
        for player_info in players_info:
            if player_info.strip():  # Check if the string is not empty
                if player_info[1].isdigit():
                    # append first two characters (shirt number) and the rest of the string (player name)
                    team_shirts.append(player_info[:2])
                    team_players.append(player_info[2:].strip())
                else:
                    # if only the first character is a digit, append only the first character (shirt number) and the rest of the string (player name)
                    team_shirts.append(player_info[0])
                    team_players.append(player_info[1:].strip())

    return team_players, team_shirts

options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--incognito")
driver = webdriver.Chrome(options=options)
driver.get("https://www.google.com")
print(driver.title)
driver.quit()
driver.get(url_str)
# wait for the page to load
driver.implicitly_wait(10)

# extract both team players and shirts
home_team_players, home_team_shirts = extract_players_info(module_sub_string=".Box.Flex.gVlxeA.jdQXvX")
away_team_players, away_team_shirts = extract_players_info(module_sub_string=".Box.Flex.gVlxeA.kaBxPK")

# iterate over all substitutes for both teams and add them
substitute_team_shirts, substitute_in_players, substitute_minute, substitute_out_players= [], [], [], []
for substitute in driver.find_elements("css selector", ".Box.Flex.deRHiB.cQgcrM"):
    # add only if the substitute is not empty and contains "Out"
    if substitute.text and "Out" in substitute.text:

        # remove any "Goal" strings from the substitute text
        substitute_text = substitute.text.replace("Goal", "").strip()

        # split ingoing and outgoing players
        in_player_str = substitute_text.split("'\nOut:  ")[0].strip()
        if in_player_str[1].isdigit():
            # append first two characters (shirt number) and the rest of the string (player name)
            substitute_team_shirts.append(in_player_str[:2])
            in_player_str = in_player_str[2:].strip()
        else:
            # if only the first character is a digit, append only the first character (shirt number) and the rest of the string (player name)
            substitute_team_shirts.append(in_player_str[0])
            in_player_str = in_player_str[1:].strip()
        substitute_in_players.append(in_player_str.split("\n")[0].strip())
        substitute_minute.append(in_player_str.split("\n")[-1].strip())

        substitute_out_players.append(substitute_text.split("'\nOut:  ")[1].strip())

# extract game date and time
game_info_element = driver.find_element("css selector", ".md\\:br_lg.pb_sm")
game_date_str = game_info_element.text.split("\n")[0].strip() # This will give you the date in the format "24/05/2025"
game_date = game_date_str.replace("/", "-") # replace the slashes with dashes to match the desired format

# extract team names
team_names_strs = []
for team_name_module in driver.find_elements("css selector", ".Text.dPnWGV"):
    team_names_strs.append(team_name_module.text)
home_team_names = [team_names_strs[0]] * len(home_team_players)
away_team_names = [team_names_strs[1]] * len(away_team_players)  

# create a dataframe for players in lineup and save as csv
team_names = home_team_names + away_team_names
players_names = home_team_players + away_team_players
shirts_numbers = home_team_shirts + away_team_shirts
lineup_df = pd.DataFrame(data={
    'Team': team_names,
    'Player': players_names,
    'Shirt Number': shirts_numbers})
lineup_df.to_csv(f"{game_date}_{team_names_strs[0]}_{team_names_strs[1]}_lineup.csv", index=False)

# create a dataframe for substitutes and save as csv
substitutes_df = pd.DataFrame(data={
    'Team': substitute_team_shirts,
    'In Player': substitute_in_players,
    'Minute': substitute_minute,
    'Out Player': substitute_out_players})
substitutes_df.to_csv(f"{game_date}_{team_names_strs[0]}_{team_names_strs[1]}_substitutes.csv", index=False)

print(f"Lineup and substitutes data saved for the game on {game_date}.")