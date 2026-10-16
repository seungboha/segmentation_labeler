"""Create two illustrative frames for trying the labeler without your own data."""
from pathlib import Path
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parent / "demo_frames"
folder.mkdir(exist_ok=True)
for index in range(2):
    image = Image.new("RGB", (960, 640), "#e4e5e8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 450, 960, 640), fill="#b9b3aa")
    shift = index * 35
    draw.ellipse((140 + shift, 145, 430 + shift, 435), fill="#bb4640", outline="#84322e", width=7)
    draw.rounded_rectangle((550 - shift, 205, 785 - shift, 445), radius=25, fill="#437db5", outline="#2b547b", width=7)
    draw.text((25, 25), f"DEMO FRAME {index + 1} - sample objects", fill="#354254")
    image.save(folder / f"frame_{index + 1:04d}.png")
print(folder)
