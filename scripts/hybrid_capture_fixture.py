"""A clearly labelled generated window for actual capture engineering checks."""

import argparse
from pathlib import Path
import time
import tkinter as tk
from PIL import Image, ImageTk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=45)
    args = parser.parse_args()
    root_path = Path(__file__).resolve().parents[1]
    root = tk.Tk()
    root.title("SCAR-X engineering capture fixture — synthetic shapes")
    root.geometry("520x390+40+40")
    root.configure(bg="white")
    tk.Label(
        root,
        text="Engineering fixture — NOT a real video benchmark",
        bg="white",
        fg="black",
    ).pack(pady=8)
    panel = tk.Label(root, bg="white")
    panel.pack()
    label = tk.Label(root, bg="white", fg="black")
    label.pack()
    started = time.monotonic()
    images = []
    for name in ("scene-first.png", "scene-second.png"):
        images.append(
            ImageTk.PhotoImage(Image.open(root_path / "local/hybrid-v1/samples" / name))
        )

    def update():
        elapsed = time.monotonic() - started
        panel.configure(image=images[int(elapsed // 8) % 2])
        label.configure(text=f"Synthetic scene · {elapsed:.1f}s")
        if elapsed >= args.seconds:
            root.destroy()
        else:
            root.after(250, update)

    update()
    root.mainloop()
