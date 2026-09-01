"""
Entry point.

Kept intentionally thin: all real logic lives in app.py / worker.py /
engine.py / config.py. Run with:

    python main.py
"""
import tkinter as tk

from app import PhysioApp


def main():
    root = tk.Tk()
    app = PhysioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
