import hikari
import re
import os
import dotenv

dotenv.load_dotenv()
BOT_TOKEN = os.getenv("TOKEN")

TWITTER_LINK_REGEX = re.compile(r"https?://(www\.)?(fxtwitter|twitter|x)\.com/\S+")

client = hikari.GatewayBot(
        intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.MESSAGE_CONTENT,
        token=BOT_TOKEN
        )

@client.listen()
async def on_started(event: hikari.StartedEvent):
    await client.update_presence(status=hikari.Status.OFFLINE)

@client.listen()
async def on_guild_message(event: hikari.GuildMessageCreateEvent):
    if TWITTER_LINK_REGEX.search(event.message.content):
        await event.message.delete()

client.run()
