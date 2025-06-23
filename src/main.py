"""
Deletes Discord messages that contain links to Twitter and attachments
containing screenshots of Twitter posts (if required modules are installed).
"""

import asyncio
import concurrent.futures
import logging
import os
import re
import sys
from typing import Optional

import aiohttp
import dotenv
import hikari

AI_AVAILABLE = True
try:
    import torch
    import torchvision.io
    import transformers
except ImportError:
    AI_AVAILABLE = False

dotenv.load_dotenv()
logger = logging.getLogger("democratizer")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TWITTER_LINK_REGEX = re.compile(r"https?://(www\.)?(twitter|x)\.com/\S+")
SCREENSHOT_MODEL = "howdyaendra/xblock-social-screenshots-2"


def process_image(media_type: str, raw_data: bytes) -> Optional[torch.Tensor]:
    """Convert raw image data to tensor in [C, H, W] format."""
    data_tensor = torch.tensor(list(raw_data), dtype=torch.uint8)
    if media_type == "image/avif":
        image = torchvision.io.decode_avif(data_tensor)
    elif media_type == "image/jpeg":
        image = torchvision.io.decode_jpeg(data_tensor)
    elif media_type == "image/png":
        image = torchvision.io.decode_png(data_tensor)
    else:
        return None

    # Convert to [C, H, W] format if needed
    if len(image.shape) == 3 and image.shape[-1] in [1, 3]:
        image = image.permute(2, 0, 1)

    # Convert grayscale to RGB
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    return image


def classify_image(processor, model, media_type: str, raw_data: bytes) -> Optional[int]:
    """Classify image and return prediction."""
    image = process_image(media_type, raw_data)
    if image is None:
        return None

    image = image.unsqueeze(0).to("cuda")
    inputs = processor(images=image / 255.0, return_tensors="pt").to("cuda")

    return model(**inputs).logits.argmax(dim=1).item()


async def should_delete_attachment(
    processor, model, attachment: hikari.Attachment
) -> bool:
    """Check if attachment should be deleted based on AI classification."""
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.proxy_url) as response:
            raw_data = await response.read()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                prediction = await loop.run_in_executor(
                    executor,
                    classify_image,
                    processor,
                    model,
                    attachment.media_type,
                    raw_data,
                )

                return prediction == 1


def main():
    """Main function to run the Discord bot."""
    client = hikari.GatewayBot(
        banner=None,
        intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.MESSAGE_CONTENT,
        token=DISCORD_TOKEN,
    )

    processor, model = None, None
    ai_enabled = "--ai" in sys.argv and AI_AVAILABLE
    if "--ai" in sys.argv and not AI_AVAILABLE:
        logger.warning("AI libraries not available. Running without AI features.")

    if ai_enabled:
        try:
            processor = transformers.AutoImageProcessor.from_pretrained(SCREENSHOT_MODEL)
            model = transformers.AutoModelForImageClassification.from_pretrained(SCREENSHOT_MODEL)
        except Exception as e:
            logger.error("Failed to load AI model: %s", e)
            ai_enabled = False

    @client.listen()
    async def on_guild_message(event: hikari.GuildMessageCreateEvent):
        if not event.message:
            return

        if event.message.content and TWITTER_LINK_REGEX.search(event.message.content):
            await event.message.delete()
            return

        if ai_enabled and event.message.attachments and processor and model:
            for attachment in event.message.attachments:
                if await should_delete_attachment(processor, model, attachment):
                    await event.message.delete()
                    break

    client.run()


if __name__ == "__main__":
    main()
