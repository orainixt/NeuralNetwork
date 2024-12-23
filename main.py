import tkinter as tk 
from window.main_gui import MainGui

def main():
    root = tk.Tk()
    app = MainGui(root) 
    root.mainloop()


if __name__ == "__main__":
    main()

    