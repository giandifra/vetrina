import cv2
import numpy as np


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
background = cv2.imread("paesaggio.png", -1)
background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
bg_h, bg_w, bg_c = background.shape

# base nera con 4 canali BGRA
base = np.zeros((bg_h, bg_w, bg_c), dtype='uint8')

# posizione in cui inserire il crop del viso
x_offset = 1786
y_offset = 650

padding = 30

# B G R
color = (255, 0, 0)
stroke = 2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    faces = face_cascade.detectMultiScale(grayFrame, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        print(x, y, w, h)
        end_cord_x = x + w
        end_cord_y = y + h
        roi_gray = grayFrame[y - padding: end_cord_y + padding, x: end_cord_x]
        roi_color = frame[y - padding: end_cord_y + padding, x: end_cord_x]

        print(roi_color.shape)
        print(base.shape)

        # save image on disk
        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        base[y_offset:y_offset + roi_color.shape[0], x_offset:x_offset + roi_color.shape[1]] = roi_color

    base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    output = overlay_transparent(base, background, 0, 0, (bg_w, bg_h))
    resize_output = cv2.resize(output, (int(output.shape[1] / 2), int(output.shape[0] / 2)))
    # cv2.imshow("frame", frame)
    cv2.imshow('image', resize_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
