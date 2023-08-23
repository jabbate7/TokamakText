# bot.py
import os

import discord
from dotenv import load_dotenv
from rag import retrieve, rag_answer_question

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
client = discord.Client(intents=intents)

from llm_interface import get_llm_interface
model = get_llm_interface('openai')

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user not in message.mentions:
        return
    question = message.content
    results = retrieve(question)
    retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
    generated_answer = rag_answer_question(question, results, model)
    display_results = True

    split_response = generated_answer.split("ANSWER:")
    thinking = split_response[0].strip()
    if len(split_response) == 1:
        answer = ""
    else:
        answer = split_response[1].strip()

    output = f"{thinking}\n\n{answer}"
    await message.channel.send(output)

client.run(TOKEN)
