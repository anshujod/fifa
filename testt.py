import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
import cloudscraper

# =========================================================
# CONFIG
# =========================================================

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

OUTPUT_DIR = "data/raw/squads_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

scraper = cloudscraper.create_scraper()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text):
    """
    Remove wikipedia references and clean spaces
    """
    text = re.sub(r"\[.*?\]", "", text)
    return text.strip()


def safe_int(x):
    """
    Convert safely to int
    """
    try:
        return int(str(x).strip())
    except:
        return None


# =========================================================
# TRANSFERMARKT MARKET VALUE SCRAPER
# =========================================================

def get_market_value(player_name):

    try:
        query = player_name.replace(" ", "+")

        url = (
            "https://www.transfermarkt.com/"
            f"schnellsuche/ergebnis/schnellsuche?query={query}"
        )

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = scraper.get(url, headers=headers)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        rows = soup.select("table.items tbody tr")

        for row in rows:

            try:
                player_cell = row.select_one("td.hauptlink")

                if not player_cell:
                    continue

                found_name = clean_text(player_cell.text)

                value_cells = row.select("td.rechts.hauptlink")

                if len(value_cells) == 0:
                    continue

                market_value = clean_text(value_cells[-1].text)

                # loose matching
                if player_name.lower() in found_name.lower():

                    return market_value

            except:
                continue

        return None

    except Exception as e:
        print(f"MV Error for {player_name}: {e}")
        return None


# =========================================================
# FETCH WIKIPEDIA PAGE
# =========================================================

response = requests.get(WIKI_URL, headers=HEADERS)

if response.status_code != 200:
    raise Exception("Failed to load Wikipedia page")

soup = BeautifulSoup(response.text, "lxml")

tables = soup.find_all("table", class_="wikitable")

print(f"Found {len(tables)} tables")

all_teams = {}

# =========================================================
# PARSE EACH TEAM
# =========================================================

for table in tables:

    # Team name from nearest heading
    heading = table.find_previous(["h2", "h3"])

    if not heading:
        continue

    team_name = clean_text(heading.text)

    # Skip weird tables
    if len(team_name) > 40:
        continue

    rows = table.find_all("tr")

    if len(rows) == 0:
        continue

    # Extract table headers
    headers = [
        clean_text(th.text).lower()
        for th in rows[0].find_all(["th", "td"])
    ]

    # Check squad table
    squad_keywords = ["pos", "player", "club"]

    if not all(any(k in h for h in headers) for k in squad_keywords):
        continue

    print(f"\nProcessing Team: {team_name}")

    players = []

    # =====================================================
    # PLAYER ROWS
    # =====================================================

    for row in rows[1:]:

        cols = row.find_all(["td", "th"])

        if len(cols) < 7:
            continue

        try:

            data = [clean_text(c.text) for c in cols]

            # Typical format:
            #
            # [No., Pos., Player, DOB(age), Caps, Goals, Club]
            #
            # Example:
            # ['1', 'GK', 'Lionel Messi', '24 June 1987 (aged 38)', '191', '112', 'Inter Miami']

            jersey_no = data[0]

            # remove jersey numbers accidentally merged
            position = re.sub(r'^\d+', '', data[1]).strip()

            name = data[2]

            dob_age = data[3]

            caps = data[4]

            goals = data[5]

            club = data[6]

            # =============================================
            # EXTRACT AGE
            # =============================================

            # Example:
            # "24 June 1987 (aged 38)"

            age_match = re.search(
                r"aged\s+(\d+)",
                dob_age.lower()
            )

            age = (
                int(age_match.group(1))
                if age_match
                else None
            )

            # =============================================
            # PLAYER OBJECT
            # =============================================

            player = {
                "name": name,
                "position": position,
                "club": club,
                "age": age,
                "caps": safe_int(caps),
                "goals": safe_int(goals),
                "market_value": None
            }

            # =============================================
            # FETCH MARKET VALUE
            # =============================================

            print(f"Fetching MV: {name}")

            mv = get_market_value(name)

            player["market_value"] = mv

            players.append(player)

            # avoid rate limiting
            time.sleep(2)

        except Exception as e:

            print(f"Row parse error: {e}")

    # Save team
    if players:

        all_teams[team_name] = players


# =========================================================
# SAVE JSON FILES
# =========================================================

for team_name, squad in all_teams.items():

    filename = (
        team_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + ".json"
    )

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            squad,
            f,
            ensure_ascii=False,
            indent=2
        )

print("\nDONE")
print(f"Saved {len(all_teams)} squads")