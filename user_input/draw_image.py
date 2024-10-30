from PIL import Image, ImageDraw

class DrawImage():

    def __init__(self):
        self.width = 1280
        self.heigth = 720
    
    def user_image_input(self): 

        image = Image.new("RGB", (self.width,self.height), "white")
        draw = ImageDraw.Draw(image)  

        input_zone_x0_y0 = (50,50)
        input_zone_x1_y1 = (200,200)
        draw.rectangle([input_zone_x0_y0,input_zone_x1_y1], fill ="blue", outline="black")
        