import os
import discord
import random
from chatbot import chat
# discord token
discord_token = 'Discord Token ID'
guild = 'chatbot server'
client= discord.Client()

'''Ready Event from discord displaying guild name and guild chatbot members'''
@client.event
async def on_ready():
    g = discord.utils.find(lambda g: g.name == guild, client.guilds)


    print(
        f'{client.user} is connected to the following guild:\n'
        f'{g.name}(id: {g.id})\n'
    )

    members = '\n - '.join([member.name for member in g.members])
    print(f'Guild Members:\n - {members}')

'''If new member join the chatbot then perform following action'''
@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hi {member.name}, welcome to my Discord server!'
    )

'''On receiving message give answer'''
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Call chat from chatbot.py
    response = chat(message.content)
    # Send answer to discord
    await message.channel.send(response)
client.run(discord_token)

