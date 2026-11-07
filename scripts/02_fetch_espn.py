
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data/raw/nba"
GAMES_DIR = OUTPUT_DIR / "games"

START_DATE = date(2025, 10, 21)
END_DATE = date(2025, 10, 23)

BASE_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba"
)


def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                request, timeout=30
            ) as response:
                return json.load(response)
        except Exception as error:
            print("Request failed:", error)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_schedule():
    games = {}
    current = START_DATE

    while current <= END_DATE:
        day = current.strftime("%Y%m%d")
        url = f"{BASE_URL}/scoreboard?dates={day}"

        print("Fetching schedule:", day)
        data = fetch_json(url)

        for event in data.get("events", []):
            event_id = event["id"]

            competition = (
                event.get("competitions") or [{}]
            )[0]

            competitors = competition.get(
                "competitors", []
            )

            teams = {
                c.get("homeAway"): c.get(
                    "team", {}
                ).get("displayName")
                for c in competitors
            }

            games[event_id] = {
                "event_id": event_id,
                "date": event.get("date"),
                "name": event.get("name"),
                "home_team": teams.get("home"),
                "away_team": teams.get("away"),
                "season_type": event.get(
                    "season", {}
                ).get("type"),
                "status": event.get(
                    "status", {}
                ).get("type", {}).get("name")
            }

        current += timedelta(days=1)
        time.sleep(0.3)

    return list(games.values())


def fetch_play_by_play(games):
    GAMES_DIR.mkdir(parents=True, exist_ok=True)

    for i, game in enumerate(games, 1):
        event_id = game["event_id"]
        output = GAMES_DIR / f"{event_id}.json"

        if output.exists():
            print(f"[{i}/{len(games)}] Cached: {event_id}")
            continue

        print(f"[{i}/{len(games)}] Fetching: {game['name']}")

        url = f"{BASE_URL}/summary?event={event_id}"

        try:
            data = fetch_json(url)
            plays = data.get("plays", [])

            if not plays:
                print("  No play-by-play available")
                continue

            with output.open("w") as f:
                json.dump(data, f)

            print("  Plays:", len(plays))

        except Exception as error:
            print("  Failed:", event_id, error)

        time.sleep(0.3)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 1: Download NBA Schedule ===")

    games = fetch_schedule()

    print("Games found:", len(games))

    schedule_path = OUTPUT_DIR / "espn_schedule.json"

    with schedule_path.open("w") as f:
        json.dump(games, f, indent=2)

    print("Schedule saved:", schedule_path)

    print("\n=== STEP 2: Download Play-by-Play ===")

    fetch_play_by_play(games)

    print("\nESPN DATA DOWNLOAD COMPLETED!")


if __name__ == "__main__":
    main()
