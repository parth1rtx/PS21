"""Generate SYNTHETIC test screenshots. Software testing only, not a research benchmark."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("ml/data/samples/screens")
OUT.mkdir(parents=True, exist_ok=True)
font = ImageFont.load_default(size=40)


def page(name: str, lines: list[str]) -> None:
    img = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        draw.text((80, y), line, fill="black", font=font)
        y += 85
    img.save(OUT / name)


page("login_paypal.png", ["PayPal", "Sign in to your account", "Email address", "Password", "Log in"])
page("login_microsoft.png", ["Microsoft", "Sign in", "Username", "Password", "Forgot password"])
page("login_generic.png", ["Welcome back", "Sign in", "Username", "Password"])
page("bakery.png", ["Sunrise Bakery", "Fresh bread every morning", "Open 7am to 5pm"])
print("written to", OUT)