import tkinter as tk

from designer_ui import DesignerUI

def main():
    root = tk.Tk()
    designer_ui = DesignerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()