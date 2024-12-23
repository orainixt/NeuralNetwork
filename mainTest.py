import tkinter as tk
from drawing.drawing_window import DrawingApp

# Créer la fenêtre principale
root = tk.Tk()

# Instancier l'application DrawingApp
app = DrawingApp(root)

# Lancer la boucle principale de l'application
root.mainloop()