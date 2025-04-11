import asyncio
import concurrent.futures
import typing
import aiohttp
import transformers
import torch
import torchvision.io
import hikari
import re
import os
import dotenv

dotenv.load_dotenv()
BOT_TOKEN = os.getenv("TOKEN")
TWITTER_LINK_REGEX = re.compile(r"https?://(www\.)?(twitter|x)\.com/\S+")

client = hikari.GatewayBot(intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.MESSAGE_CONTENT, token=BOT_TOKEN)
processor = transformers.AutoImageProcessor.from_pretrained("howdyaendra/xblock-social-screenshots-1")
model = transformers.AutoModelForImageClassification.from_pretrained("howdyaendra/xblock-social-screenshots-1").to("cuda")

def get_image(media_type: str, raw_data: bytes) -> typing.Optional[torch.Tensor]:
    data_tensor = torch.tensor(list(raw_data), dtype=torch.uint8)
    if media_type == "image/avif":
        image = torchvision.io.decode_avif(data_tensor)
    elif media_type == "image/jpeg":
        image = torchvision.io.decode_jpeg(data_tensor)
    elif media_type == "image/png":
        image = torchvision.io.decode_png(data_tensor)
    else:
        return None

    # Ensure the image is in [C, H, W] format
    if len(image.shape) == 3 and image.shape[-1] in [1, 3]:  # Check if it's [H, W, C]
        image = image.permute(2, 0, 1)  # Convert [H, W, C] to [C, H, W]

    # Handle grayscale images by converting them to RGB
    if image.shape[0] == 1:  # Grayscale images have one channel
        image = image.repeat(3, 1, 1)  # Convert [1, H, W] to [3, H, W]

    return image

def run_processor(media_type: str, raw_data: bytes) -> typing.Optional[int]:
    # Get the processed image
    image = get_image(media_type, raw_data)
    if image is None:
        return None

    # Add a batch dimension and move to CUDA
    image = image.unsqueeze(0).to("cuda")  # Shape: [1, C, H, W]

    # Preprocess the image using the Hugging Face processor
    inputs = processor(images=image / 255.0, return_tensors="pt").to("cuda")  # Normalize pixel values to [0, 1]

    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits

    return logits.argmax(dim=1).item()

async def should_delete(attachment: hikari.Attachment) -> bool:
    loop = asyncio.get_running_loop()
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.proxy_url) as response:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                prediction = await loop.run_in_executor(
                    executor,
                    run_processor,
                    attachment.media_type,
                    await response.read()
                )
                if prediction == 1:
                    return True
    return False

@client.listen()
async def on_started(event: hikari.StartedEvent):
    await client.update_presence(status=hikari.Status.OFFLINE)

@client.listen()
async def on_guild_message(event: hikari.GuildMessageCreateEvent):
    if event is None or event.message is None:
        return

    if event.message.content is not None and TWITTER_LINK_REGEX.search(event.message.content):
        await event.message.delete()

    if event.message.attachments is not None:
        for attachment in event.message.attachments:
            if await should_delete(attachment):
                await event.message.delete()
                break

client.run()

