from mcp.server.fastmcp import FastMCP
import datetime
import hashlib
import random
import json
import os
import urllib.parse

mcp = FastMCP("daily-life")


# ─────────────────────────────────────────────
# TOOL 1: Daily Horoscope
# ─────────────────────────────────────────────
@mcp.tool()
def horoscope(zodiac_sign: str) -> str:
    """
    Get your daily horoscope!
    zodiac_sign options: aries, taurus, gemini, cancer, leo, virgo,
                         libra, scorpio, sagittarius, capricorn, aquarius, pisces
    Refreshes automatically every day.
    """
    signs = ["aries", "taurus", "gemini", "cancer", "leo", "virgo",
             "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]

    sign = zodiac_sign.lower().strip()
    if sign not in signs:
        return f"❌ Invalid sign. Choose from: {', '.join(signs)}"

    today = datetime.date.today().isoformat()
    seed = int(hashlib.sha256(f"{sign}{today}".encode()).hexdigest(), 16)
    rng = random.Random(seed)

    sign_symbols = {
        "aries": "♈", "taurus": "♉", "gemini": "♊", "cancer": "♋",
        "leo": "♌", "virgo": "♍", "libra": "♎", "scorpio": "♏",
        "sagittarius": "♐", "capricorn": "♑", "aquarius": "♒", "pisces": "♓"
    }

    moods = ["Energetic", "Calm", "Creative", "Reflective", "Bold",
             "Compassionate", "Focused", "Playful", "Determined", "Dreamy"]

    lucky_colors = ["Royal Blue", "Emerald Green", "Sunset Orange", "Rose Gold",
                    "Pearl White", "Midnight Purple", "Coral Red", "Sky Blue",
                    "Golden Yellow", "Forest Green", "Crimson", "Lavender"]

    lucky_numbers = rng.sample(range(1, 50), 3)

    predictions = [
        "A surprise opportunity will come your way — stay open to new paths.",
        "Someone from your past may reach out. Embrace the reconnection.",
        "Your creativity is at its peak today. Express yourself boldly.",
        "Financial decisions made today will have long-lasting positive effects.",
        "A conversation with a friend will give you the clarity you have been seeking.",
        "Trust your instincts — they are sharper than usual today.",
        "An unexpected gift or compliment will brighten your afternoon.",
        "Take a moment of silence today. The answers you seek are within.",
        "A challenge at work or study will turn into your biggest achievement.",
        "Today favors travel plans, even short ones. Explore something new.",
        "Your patience will be tested, but the reward at the end is worth it.",
        "A new friendship is forming. Be warm and welcoming to strangers.",
    ]

    love = [
        "Singles: Someone is noticing you more than you realise.",
        "Couples: A small romantic gesture today will mean the world.",
        "Singles: Focus on self-love today — it attracts the right energy.",
        "Couples: Clear communication will strengthen your bond.",
        "Singles: Your charm is magnetic today. Go out and shine.",
        "Couples: Plan something spontaneous together — even a walk counts.",
    ]

    career = [
        "A mentor or senior figure will offer valuable advice. Listen carefully.",
        "Your hard work is being noticed behind the scenes.",
        "A collaboration opportunity is on the horizon.",
        "Focus on one task at a time — multitasking will slow you down today.",
        "Your ideas in a meeting or discussion will stand out.",
        "A deadline you have been dreading is actually more manageable than you think.",
    ]

    health = [
        "Drink more water today — your body is asking for it.",
        "A short walk outside will do wonders for your mood.",
        "Rest is productive. Do not feel guilty for taking a break.",
        "Stretching for 5 minutes in the morning will boost your energy.",
        "Watch your diet today — your stomach will thank you.",
        "Spend time in sunlight. Even 15 minutes makes a difference.",
    ]

    symbol = sign_symbols[sign]
    lucky_color = rng.choice(lucky_colors)
    mood = rng.choice(moods)
    intensity = rng.randint(60, 99)

    return (
        f"{symbol} DAILY HOROSCOPE — {sign.upper()}\n"
        f"{'='*44}\n"
        f"  Date         : {today}\n"
        f"  Mood         : {mood}\n"
        f"  Intensity    : {intensity}%\n"
        f"  Lucky Color  : {lucky_color}\n"
        f"  Lucky Numbers: {', '.join(map(str, lucky_numbers))}\n"
        f"{'─'*44}\n"
        f"  TODAY'S PREDICTION:\n"
        f"  {rng.choice(predictions)}\n\n"
        f"  LOVE:\n"
        f"  {rng.choice(love)}\n\n"
        f"  CAREER & STUDY:\n"
        f"  {rng.choice(career)}\n\n"
        f"  HEALTH:\n"
        f"  {rng.choice(health)}\n"
        f"{'='*44}\n"
        f"  (Refreshes automatically at midnight)"
    )


# ─────────────────────────────────────────────
# TOOL 2: Currency Converter
# ─────────────────────────────────────────────
@mcp.tool()
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Converts currency between major world currencies.
    Supported: USD, EUR, GBP, INR, JPY, AUD, CAD, CHF, CNY, SGD, AED, SAR
    Example: amount=100, from_currency="USD", to_currency="INR"
    """
    rates_to_usd = {
        "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "INR": 0.012,
        "JPY": 0.0067, "AUD": 0.65, "CAD": 0.74, "CHF": 1.13,
        "CNY": 0.138, "SGD": 0.75, "AED": 0.272, "SAR": 0.267,
    }
    currency_names = {
        "USD": "US Dollar", "EUR": "Euro", "GBP": "British Pound",
        "INR": "Indian Rupee", "JPY": "Japanese Yen", "AUD": "Australian Dollar",
        "CAD": "Canadian Dollar", "CHF": "Swiss Franc", "CNY": "Chinese Yuan",
        "SGD": "Singapore Dollar", "AED": "UAE Dirham", "SAR": "Saudi Riyal"
    }
    currency_symbols = {
        "USD": "$", "EUR": "€", "GBP": "£", "INR": "Rs",
        "JPY": "Y", "AUD": "A$", "CAD": "C$", "CHF": "Fr",
        "CNY": "Y", "SGD": "S$", "AED": "AED", "SAR": "SAR"
    }

    from_c = from_currency.upper().strip()
    to_c = to_currency.upper().strip()

    if from_c not in rates_to_usd:
        return f"❌ Unknown currency: {from_c}. Supported: {', '.join(rates_to_usd.keys())}"
    if to_c not in rates_to_usd:
        return f"❌ Unknown currency: {to_c}. Supported: {', '.join(rates_to_usd.keys())}"

    amount_in_usd = amount * rates_to_usd[from_c]
    converted = amount_in_usd / rates_to_usd[to_c]
    rate = rates_to_usd[from_c] / rates_to_usd[to_c]

    from_sym = currency_symbols[from_c]
    to_sym = currency_symbols[to_c]

    return (
        f"CURRENCY CONVERTER\n"
        f"{'='*40}\n"
        f"  {from_sym}{amount:,.2f} {from_c}  =>  {to_sym}{converted:,.2f} {to_c}\n"
        f"{'─'*40}\n"
        f"  From : {currency_names[from_c]} ({from_c})\n"
        f"  To   : {currency_names[to_c]} ({to_c})\n"
        f"  Rate : 1 {from_c} = {rate:.4f} {to_c}\n"
        f"  Date : {datetime.date.today()}\n"
        f"{'─'*40}\n"
        f"  Note: Rates are approximate.\n"
        f"  For exact rates check xe.com"
    )


# ─────────────────────────────────────────────
# TOOL 3: Smart Reminder & To-Do Manager
# ─────────────────────────────────────────────
@mcp.tool()
def reminder(action: str, task: str = "", priority: str = "medium") -> str:
    """
    Your personal daily reminder and to-do manager.
    action options:
      add   - add a new reminder (needs task + optional priority: low/medium/high)
      list  - show all pending reminders
      done  - mark a task as done (use task name or number from list)
      clear - clear all completed tasks
    Example: action="add", task="Call mom at 5pm", priority="high"
    """
    path = "reminders.json"
    data = {"pending": [], "done": []}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    action = action.lower().strip()
    priority = priority.lower().strip()
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now().strftime("%H:%M")
    priority_icons = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}
    p_icon = priority_icons.get(priority, "[MED]")

    if action == "add":
        if not task:
            return "❌ Please provide a task. Example: action='add', task='Buy groceries'"
        item = {"id": len(data["pending"]) + len(data["done"]) + 1,
                "task": task, "priority": priority, "added": f"{today} {now}"}
        data["pending"].append(item)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return (
            f"✅ Reminder added!\n"
            f"  {p_icon} {task}\n"
            f"  Added: {today} at {now}\n"
            f"  Total pending: {len(data['pending'])}"
        )

    elif action == "list":
        if not data["pending"]:
            return "No pending reminders! You're all caught up."
        lines = [f"YOUR REMINDERS — {today}", "─" * 36]
        for i, item in enumerate(data["pending"], 1):
            p = item['priority'].upper()
            lines.append(f"  {i}. [{p}] {item['task']}")
            lines.append(f"     Added: {item['added']}")
        lines.append(f"\n  Total: {len(data['pending'])} pending | {len(data['done'])} done")
        return "\n".join(lines)

    elif action == "done":
        if not task:
            return "❌ Specify which task. Use task name or number from list."
        matched = None
        if task.isdigit():
            idx = int(task) - 1
            if 0 <= idx < len(data["pending"]):
                matched = data["pending"][idx]
        else:
            for item in data["pending"]:
                if task.lower() in item["task"].lower():
                    matched = item
                    break
        if not matched:
            return f"❌ Could not find: '{task}'. Use action='list' to see tasks."
        data["pending"].remove(matched)
        matched["completed"] = f"{today} {now}"
        data["done"].append(matched)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return (
            f"✅ Task completed!\n"
            f"  {matched['task']}\n"
            f"  Done: {today} at {now}\n"
            f"  Remaining: {len(data['pending'])} tasks"
        )

    elif action == "clear":
        cleared = len(data["done"])
        data["done"] = []
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return f"Cleared {cleared} completed tasks."

    else:
        return "❌ Invalid action. Use: add / list / done / clear"


# ─────────────────────────────────────────────
# TOOL 4: Location Info & Map Link Generator
# ─────────────────────────────────────────────
@mcp.tool()
def location_info(city: str, country: str = "") -> str:
    """
    Get useful info about any city + a Google Maps link.
    Example: city="Hyderabad", country="India"
             city="Tokyo", country="Japan"
    """
    cities = {
        "hyderabad": {"country": "India", "tz": "IST (UTC+5:30)", "pop": "10.5 million",
                      "known_for": "Biryani, Charminar, IT hub, Pearls",
                      "language": "Telugu, Urdu, Hindi",
                      "fact": "Hyderabad is called the City of Pearls and Cyberabad for its tech industry."},
        "mumbai": {"country": "India", "tz": "IST (UTC+5:30)", "pop": "20.7 million",
                   "known_for": "Bollywood, Gateway of India, street food",
                   "language": "Hindi, Marathi",
                   "fact": "Mumbai handles over 70% of India's maritime trade."},
        "delhi": {"country": "India", "tz": "IST (UTC+5:30)", "pop": "32 million",
                  "known_for": "Red Fort, India Gate, street food, history",
                  "language": "Hindi, Punjabi, Urdu",
                  "fact": "Delhi is one of the most populated cities in the world."},
        "bangalore": {"country": "India", "tz": "IST (UTC+5:30)", "pop": "13 million",
                      "known_for": "Silicon Valley of India, gardens, startups",
                      "language": "Kannada, English",
                      "fact": "Bangalore has the highest number of tech startups in India."},
        "london": {"country": "UK", "tz": "GMT/BST (UTC+0/+1)", "pop": "9 million",
                   "known_for": "Big Ben, Tower Bridge, royalty, theatre",
                   "language": "English",
                   "fact": "London has over 170 museums and is one of the most visited cities on Earth."},
        "new york": {"country": "USA", "tz": "EST/EDT (UTC-5/-4)", "pop": "8.3 million",
                     "known_for": "Times Square, Statue of Liberty, Wall Street",
                     "language": "English",
                     "fact": "New York City has over 800 languages spoken — the most diverse city on Earth."},
        "tokyo": {"country": "Japan", "tz": "JST (UTC+9)", "pop": "13.9 million",
                  "known_for": "Technology, anime, sushi, cherry blossoms",
                  "language": "Japanese",
                  "fact": "Tokyo has the world's busiest pedestrian crossing at Shibuya."},
        "dubai": {"country": "UAE", "tz": "GST (UTC+4)", "pop": "3.5 million",
                  "known_for": "Burj Khalifa, luxury, gold souk, desert safaris",
                  "language": "Arabic, English",
                  "fact": "Dubai went from a fishing village to a global city in under 50 years."},
        "paris": {"country": "France", "tz": "CET/CEST (UTC+1/+2)", "pop": "2.1 million",
                  "known_for": "Eiffel Tower, fashion, cuisine, art",
                  "language": "French",
                  "fact": "Paris has only one stop sign in the entire city."},
        "singapore": {"country": "Singapore", "tz": "SGT (UTC+8)", "pop": "5.9 million",
                      "known_for": "Marina Bay Sands, cleanliness, food paradise",
                      "language": "English, Malay, Mandarin, Tamil",
                      "fact": "Singapore is one of only three city-states in the world."},
        "sydney": {"country": "Australia", "tz": "AEDT (UTC+11)", "pop": "5.3 million",
                   "known_for": "Opera House, Harbour Bridge, beaches",
                   "language": "English",
                   "fact": "Sydney is not Australia's capital — that is Canberra."},
        "chennai": {"country": "India", "tz": "IST (UTC+5:30)", "pop": "7.1 million",
                    "known_for": "Marina Beach, temples, classical music, filter coffee",
                    "language": "Tamil",
                    "fact": "Chennai has the second longest urban beach in the world — Marina Beach."},
    }

    city_key = city.lower().strip()
    maps_query = urllib.parse.quote(f"{city} {country}".strip())
    maps_link = f"https://maps.google.com/?q={maps_query}"

    if city_key in cities:
        info = cities[city_key]
        return (
            f"LOCATION INFO — {city.upper()}\n"
            f"{'='*44}\n"
            f"  Country    : {info['country']}\n"
            f"  Timezone   : {info['tz']}\n"
            f"  Population : {info['pop']}\n"
            f"  Language   : {info['language']}\n"
            f"  Known For  : {info['known_for']}\n"
            f"{'─'*44}\n"
            f"  FUN FACT:\n"
            f"  {info['fact']}\n"
            f"{'─'*44}\n"
            f"  Google Maps:\n"
            f"  {maps_link}"
        )
    else:
        return (
            f"LOCATION INFO — {city.upper()}\n"
            f"{'='*44}\n"
            f"  City not in local database.\n"
            f"  Here is your Google Maps link:\n\n"
            f"  {maps_link}\n"
            f"{'─'*44}\n"
            f"  Cities in database:\n"
            f"  Hyderabad, Mumbai, Delhi, Bangalore,\n"
            f"  Chennai, London, New York, Tokyo,\n"
            f"  Dubai, Paris, Singapore, Sydney"
        )


# ─────────────────────────────────────────────
# TOOL 5: Daily Habit Tracker
# ─────────────────────────────────────────────
@mcp.tool()
def habit_tracker(action: str, habit: str = "", value: str = "") -> str:
    """
    Track your daily habits!
    action options: log / view / streak
    habit options: water, exercise, sleep, reading, meditation, steps
    value examples: "8 glasses", "30 minutes", "7 hours", "20 pages"
    Example: action="log", habit="water", value="8 glasses"
             action="view"
             action="streak"
    """
    path = "habits.json"
    today = datetime.date.today().isoformat()
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    action = action.lower().strip()
    habit = habit.lower().strip()

    habit_icons = {
        "water": "Water", "exercise": "Exercise", "sleep": "Sleep",
        "reading": "Reading", "meditation": "Meditation", "steps": "Steps"
    }
    habit_goals = {
        "water": "8 glasses", "exercise": "30 minutes", "sleep": "8 hours",
        "reading": "20 pages", "meditation": "10 minutes", "steps": "8000 steps"
    }

    if action == "log":
        if not habit or not value:
            return "❌ Provide habit and value. Example: action='log', habit='water', value='8 glasses'"
        if today not in data:
            data[today] = {}
        data[today][habit] = {"value": value, "time": datetime.datetime.now().strftime("%H:%M")}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        goal = habit_goals.get(habit, "")
        return (
            f"HABIT LOGGED!\n"
            f"  Habit : {habit.capitalize()}\n"
            f"  Value : {value}\n"
            f"  Goal  : {goal}\n"
            f"  Date  : {today}\n"
            f"  Keep it up! Consistency is the key."
        )

    elif action == "view":
        if today not in data or not data[today]:
            return f"No habits logged for today ({today}) yet.\nUse action='log' to start!"
        today_data = data[today]
        lines = [f"TODAY'S HABITS — {today}", "─" * 36]
        for h in habit_icons:
            if h in today_data:
                lines.append(f"  [DONE] {h.capitalize()}: {today_data[h]['value']} at {today_data[h]['time']}")
            else:
                lines.append(f"  [    ] {h.capitalize()}: not logged (goal: {habit_goals[h]})")
        logged = len(today_data)
        total = len(habit_icons)
        lines.append(f"\n  Progress: {logged}/{total} habits done today")
        if logged == total:
            lines.append("  PERFECT DAY! All habits completed!")
        return "\n".join(lines)

    elif action == "streak":
        if not data:
            return "No habit data yet. Start with action='log'!"
        all_dates = sorted(data.keys(), reverse=True)
        lines = ["YOUR HABIT STREAKS", "─" * 36]
        for h in habit_icons:
            streak = 0
            for d in all_dates:
                if h in data.get(d, {}):
                    streak += 1
                else:
                    break
            lines.append(f"  {h.capitalize()}: {streak} day streak")
        lines.append(f"\n  Total days tracked: {len(all_dates)}")
        return "\n".join(lines)

    else:
        return "❌ Invalid action. Use: log / view / streak"


# ─────────────────────────────────────────────
# TOOL 6: Age & Date Calculator
# ─────────────────────────────────────────────
@mcp.tool()
def date_calculator(mode: str, date1: str, date2: str = "") -> str:
    """
    A powerful date and age calculator.
    mode options:
      age        - calculate exact age (date1=YYYY-MM-DD)
      countdown  - days until a future event (date1=YYYY-MM-DD)
      difference - days between two dates (date1 and date2=YYYY-MM-DD)
      dayofweek  - what day of the week is a date (date1=YYYY-MM-DD)
    Example: mode="age", date1="2000-05-20"
             mode="countdown", date1="2025-12-25"
             mode="difference", date1="2024-01-01", date2="2025-01-01"
    """
    try:
        d1 = datetime.date.fromisoformat(date1)
    except ValueError:
        return "❌ Invalid date1. Use YYYY-MM-DD format (e.g. 2000-05-20)"

    today = datetime.date.today()
    mode = mode.lower().strip()

    if mode == "age":
        if d1 > today:
            return "❌ Birthdate cannot be in the future."
        years = today.year - d1.year - ((today.month, today.day) < (d1.month, d1.day))
        months = (today.month - d1.month) % 12
        days_lived = (today - d1).days
        hours_lived = days_lived * 24
        next_bday = d1.replace(year=today.year)
        if next_bday < today:
            next_bday = d1.replace(year=today.year + 1)
        days_to_bday = (next_bday - today).days
        return (
            f"AGE CALCULATOR\n"
            f"{'='*36}\n"
            f"  Birthdate    : {date1}\n"
            f"  Today        : {today}\n"
            f"{'─'*36}\n"
            f"  Age          : {years} years, {months} months\n"
            f"  Days lived   : {days_lived:,}\n"
            f"  Hours lived  : {hours_lived:,}\n"
            f"  Next birthday: in {days_to_bday} days ({next_bday})\n"
            f"{'='*36}"
        )

    elif mode == "countdown":
        if d1 < today:
            return f"That date has already passed ({(today - d1).days} days ago)."
        days_left = (d1 - today).days
        weeks = days_left // 7
        day_name = d1.strftime("%A")
        return (
            f"COUNTDOWN\n"
            f"{'='*36}\n"
            f"  Target date : {date1} ({day_name})\n"
            f"  Today       : {today}\n"
            f"{'─'*36}\n"
            f"  Days left   : {days_left}\n"
            f"  Weeks left  : {weeks} weeks, {days_left % 7} days\n"
            f"{'='*36}"
        )

    elif mode == "difference":
        if not date2:
            return "❌ Please provide date2. Example: date2='2025-12-31'"
        try:
            d2 = datetime.date.fromisoformat(date2)
        except ValueError:
            return "❌ Invalid date2. Use YYYY-MM-DD format"
        diff = abs((d2 - d1).days)
        return (
            f"DATE DIFFERENCE\n"
            f"{'='*36}\n"
            f"  Date 1 : {date1}\n"
            f"  Date 2 : {date2}\n"
            f"{'─'*36}\n"
            f"  Difference : {diff} days\n"
            f"  In weeks   : {diff // 7} weeks, {diff % 7} days\n"
            f"  In months  : ~{round(diff/30.44, 1)} months\n"
            f"  In years   : ~{round(diff/365.25, 2)} years\n"
            f"{'='*36}"
        )

    elif mode == "dayofweek":
        day_name = d1.strftime("%A, %d %B %Y")
        diff = (d1 - today).days
        if diff == 0:
            rel = "Today"
        elif diff == 1:
            rel = "Tomorrow"
        elif diff == -1:
            rel = "Yesterday"
        elif diff > 0:
            rel = f"{diff} days from now"
        else:
            rel = f"{abs(diff)} days ago"
        return (
            f"DAY OF WEEK\n"
            f"{'='*36}\n"
            f"  Date     : {date1}\n"
            f"  Day      : {day_name}\n"
            f"  Relative : {rel}\n"
            f"{'='*36}"
        )

    else:
        return "❌ Invalid mode. Use: age / countdown / difference / dayofweek"


if __name__ == "__main__":
    mcp.run()