import tkinter as tk 
from tkinter import filedialog
from PIL import Image, ImageDraw

class DrawingApp : 

    def __init__(self,root):
        self.root = root 
        self.root.title("Draw a Number :")
        #parameter of the canvas
        self.canvas_width = 1280 
        self.canvas_height = 720 
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack() 
        #initialize image et draw object 
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), color="white")
        self.draw = ImageDraw.Draw(self.image) 
        #x/y-axis to handle mouse 
        self.last_x = None 
        self.last_y = None 
        #save button 
        save_button = tk.Button(root,text="Save File",command=self.save_image)
        #binding controls 
        self.canvas.bind("<Button-1>",self.save_to_start) 
        self.canvas.bind("<B1-Motion>",self.draw_line) 
        self.canvas.bing("<ButtonRealease-1>",self.stop_drawing)

    def save_to_start(self,event):
        """
        Function used to save the last position of x and y when user start drawing
            """
        self.last_x = event.x 
        self.last_y = event.y

    def draw_line(self, event): 
        """
        Function used to draw a line between 
        """
        #check if there is valid coordinates
        if self.last_x and self.last_y: 
            x = event.x
            y = event.y
            self.canvas.create_line(self.last_x,self.last_y,x,y,width=2,fill="black",capstyle=tk.ROUND,smooth=True)
            self.draw.line([self.last_x,self.last_y,x,y],fill="black",width=2)
            self.last_x = x 
            self.last_y = y
        
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.image.save(file_path)
            print(f"Image enregistrée sous {file_path}")

