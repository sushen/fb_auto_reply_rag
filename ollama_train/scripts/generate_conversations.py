import json
import random
from pathlib import Path

BINANCE_LINK = "https://www.binance.com/register?ref=35023868"
DISCORD_LINK = "https://discord.com/invite/GHjeawSYcp"

greetings = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening"
]

interests = [
    "I want to trade",
    "tell me about trading ai",
    "how does the bot work",
    "I want to earn",
    "I want to learn algorithmic trading"
]

assistant_openings = [
    "Hello! Welcome to Trading AI. We help people automate Binance trading using our AI trading bot.",
    "Hi! We help traders use AI-powered bots to trade automatically on Binance.",
    "Welcome! Trading AI allows users to automate Binance trading using algorithmic strategies."
]

bot_explanations = [
    "Our Trading AI bot connects to your Binance account using API keys and automatically executes trades.",
    "The Trading AI system analyzes market data and performs trades automatically on Binance.",
    "Instead of manual trading, the bot follows algorithmic strategies and executes trades automatically."
]

affiliate_messages = [
    "Many users also earn through our affiliate program by introducing others to Trading AI.",
    "You can also earn commissions by sharing Trading AI with other traders.",
    "Our affiliate system allows users to earn by helping others start automated trading."
]

learning_messages = [
    "We also teach algorithmic trading using Python so users can understand how trading bots work.",
    "Our training program teaches Python and real algorithmic trading strategies.",
    "Students learn how to build automated trading strategies using Python."
]

dataset = []

for i in range(400):

    greeting = random.choice(greetings)
    opening = random.choice(assistant_openings)
    interest = random.choice(interests)

    explanation = random.choice(bot_explanations)
    affiliate = random.choice(affiliate_messages)
    learning = random.choice(learning_messages)

    conversation = [
        {"role": "user", "content": greeting},
        {"role": "assistant", "content": opening},

        {"role": "user", "content": interest},
        {"role": "assistant", "content": explanation},

        {"role": "assistant", "content": learning},
        {"role": "assistant", "content": affiliate},

        {
            "role": "assistant",
            "content": f"You can start by creating your Binance account here: {BINANCE_LINK}"
        },

        {
            "role": "assistant",
            "content": f"After creating your account, please join our Discord community to request your Trading AI bot license: {DISCORD_LINK}"
        }
    ]

    dataset.append({
        "conversation_id": i + 1,
        "messages": conversation
    })

base_dir = Path(__file__).resolve().parent.parent
output_path = base_dir / "dataset" / "trading_ai_conversations.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} focused conversations at: {output_path}")