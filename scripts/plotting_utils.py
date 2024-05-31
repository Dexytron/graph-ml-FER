import cv2 as cv
import numpy as np

# colors
# values =(blue, green, red) opencv accepts BGR values not RGB
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
points_list =[(200, 300), (150, 150), (400, 200)]
def drawColor(img, colors):
    x, y = 0,10
    w, h = 20, 30

    for color in colors:
        x += w+5
        # y += 10
        cv.rectangle(img, (x-6, y-5 ), (x+w+5, y+h+5), (10, 50, 10), -1)
        cv.rectangle(img, (x, y ), (x+w, y+h), color, -1)


def textWithBackground(img, text, font, fontScale, textPos, textThickness=1, textColor=(255, 255, 255), bgColor=(0, 0, 0), bgOpacity=0.5, corner='top-left'):
    """
    Draws text with a rounded translucent background box and padding.
    @param img: (mat) The image on which you want to draw text.
    @param text: (string) The text you want to draw.
    @param font: Font face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN, etc.
    @param fontScale: (double) The size of the text, how big it should be.
    @param textPos: tuple(x,y) Padding between the outer frame and the text box.
    @param textThickness: (int) Font weight, how bold it should be.
    @param textColor: tuple(BGR) Color values for text (0 to 255 each).
    @param bgColor: tuple(BGR) Color values for background (0 to 255 each).
    @param bgOpacity: float (0 to 1.0) Transparency of the text background.
    @param corner: string Reference corner for positioning text ("top-left", "top-right", "bottom-left", "bottom-right").
    @return: img (mat) with text drawn on it.
    """
    (t_w, t_h), _ = cv.getTextSize(text, font, fontScale, textThickness)
    pad_x = int(fontScale * 10)  # Padding proportional to font scale
    pad_y = int(fontScale * 10)  # Padding proportional to font scale

    img_h, img_w = img.shape[:2]

    if corner == 'top-left':
        x, y = textPos
        x += pad_x
        y += t_h + pad_y
    elif corner == 'top-right':
        x, y = img_w - textPos[0] - t_w - pad_x, textPos[1] + t_h + pad_y
    elif corner == 'bottom-left':
        x, y = textPos[0] + pad_x, img_h - textPos[1] - pad_y
    elif corner == 'bottom-right':
        x, y = img_w - textPos[0] - t_w - pad_x, img_h - textPos[1] - pad_y

    overlay = img.copy()

    # Calculate box dimensions with padding
    box_x1, box_y1 = x - pad_x, y + pad_y
    box_x2, box_y2 = x + t_w + pad_x, y - t_h - pad_y

    # Draw a rounded rectangle (background box)
    radius = int(min(t_w, t_h) / 2)
    color = (bgColor[0], bgColor[1], bgColor[2])

    cv.rectangle(overlay, (box_x1 + radius, box_y1), (box_x2 - radius, box_y2), color, -1)
    cv.rectangle(overlay, (box_x1, box_y1 - radius), (box_x2, box_y2 + radius), color, -1)
    cv.circle(overlay, (box_x1 + radius, box_y1 - radius), radius, color, -1)
    cv.circle(overlay, (box_x2 - radius, box_y1 - radius), radius, color, -1)
    cv.circle(overlay, (box_x1 + radius, box_y2 + radius), radius, color, -1)
    cv.circle(overlay, (box_x2 - radius, box_y2 + radius), radius, color, -1)

    # Blend the overlay with the original image
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0)

    # Draw the text on the image
    cv.putText(new_img, text, (x, y), font, fontScale, textColor, textThickness)

    return new_img


def textBlurBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0),kneral=(33,33) , pad_x=3, pad_y=3):
    """
    Draw text with background blured,  control the blur value, with kernal(odd, odd)
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param kneral: tuple(3,3) int as odd number:  higher the value, more blurry background would be
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels)  padding of in y direction
    @return: img mat, with text drawn, with background blured

    call the function:
     img =textBlurBackground(img, 'Blured Background Text', cv2.FONT_HERSHEY_COMPLEX, 0.9, (20, 60),2, (0,255, 0), (49,49), 13, 13 )
    """

    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    blur_roi = img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x] # croping Text Background
    img[y-pad_y-t_h: y+pad_y, x-pad_x:x+t_w+pad_x]=cv.blur(blur_roi, kneral)  # merging the blured background to img
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness )
    # cv.imshow('blur roi', blur_roi)
    # cv.imshow('blured', img)

    return img

def fillPolyTrans(img, points, color, opacity):
    """
    @param img: (mat) input image, where shape is drawn.
    @param points: list [tuples(int, int) these are the points custom shape,FillPoly
    @param color: (tuples (int, int, int)
    @param opacity:  it is transparency of image.
    @return: img(mat) image with rectangle draw.

    """
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv.polylines(img, [list_to_np_array], True, color,1, cv.LINE_AA)
    return img

def rectTrans(img, pt1, pt2, color, thickness, opacity):
    """

    @param img: (mat) input image, where shape is drawn.
    @param pt1: tuple(int,int) it specifies the starting point(x,y) os rectangle
    @param pt2: tuple(int,int)  it nothing but width and height of rectangle
    @param color: (tuples (int, int, int), it tuples of BGR values
    @param thickness: it thickness of board line rectangle, if (-1) passed then rectangle will be fulled with color.
    @param opacity:  it is transparency of image.
    @return:
    """
    overlay = img.copy()
    cv.rectangle(overlay, pt1, pt2, color, thickness)
    new_img = cv.addWeighted(overlay, opacity, img, 1 - opacity, 0) # overlaying the rectangle on the image.
    img = new_img

    return img

def main():
    cap = cv.VideoCapture('Girl.mp4')
    counter =0
    while True:
        success, img = cap.read()
        # img = np.zeros((1000,1000, 3), dtype=np.uint8)
        img=rectTrans(img, pt1=(30, 320), pt2=(160, 260), color=(0,255,255),thickness=-1, opacity=0.6)
        img =fillPolyTrans(img=img, points=points_list, color=(0,255,0), opacity=.5)
        drawColor(img, [BLACK,WHITE ,BLUE,RED,CYAN,YELLOW,MAGENTA,GRAY ,GREEN,PURPLE,ORANGE,PINK])
        textBlurBackground(img, 'Blured Background Text', cv.FONT_HERSHEY_COMPLEX, 0.8, (60, 140),2, YELLOW, (71,71), 13, 13)
        img=textWithBackground(img, 'Colored Background Texts', cv.FONT_HERSHEY_SIMPLEX, 0.8, (60,80), textThickness=2, bgColor=GREEN, textColor=BLACK, bgOpacity=0.7, pad_x=6, pad_y=6)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cv.imwrite('color_image.png', img)
        counter +=1
        cv.imshow('img', img)
        cv.imwrite(f'image/image_{counter}.png', img)
        if cv.waitKey(1) ==ord('q'):
            break

if __name__ == "__main__":
    main()