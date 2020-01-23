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
background = cv2.imread("memoria-3-bucato.png", cv2.IMREAD_UNCHANGED)
background_base = cv2.imread("memoria-3-base.png", cv2.IMREAD_UNCHANGED)
background_2 = cv2.imread("memoria-3-bucato-2.png", -1)
background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
red_level = cv2.imread("memoria-3-rosso.png", cv2.IMREAD_UNCHANGED)
bg_h, bg_w, bg_c = background.shape
print(background.shape)
# base nera con 4 canali BGRA


# posizione in cui inserire il crop del viso
# x_offset = 1786
# y_offset = 650

x_offset = 562
y_offset = 160

# empty_width = 84
# empty_height = 100
# empty_center_x = x_offset + empty_width / 2
# empty_center_y = y_offset + empty_height / 2

empty_h = 150
empty_w = 110
empty_center_x = 678
empty_center_y = 182
print(empty_center_x)
print(empty_center_y)

padding = 30

# B G R
blue_color = (255, 0, 0)
red_color = (0, 0, 255)
stroke = 2

cap = cv2.VideoCapture(0)

'''
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

capture_buffer = np.empty(shape=(int(h), int(w), 4), dtype=np.uint8)
'''

# cap.set(cv2.CAP_PROP_FPS, 30)

while True:

    # 1280x720
    ret, frame = cap.read()
    base = np.zeros((bg_h, bg_w, bg_c), dtype='uint8')
    scale = background.shape[1] / frame.shape[1]
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame2 = cv2.resize(frame, (background.shape[1], background.shape[0] * scale))
    grayFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame2, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        start_cord_x = x
        start_cord_y = y - padding
        end_cord_x = x + w
        new_h = h + padding * 2
        end_cord_y = start_cord_y + new_h
        # roi_gray = grayFrame2[start_cord_y: end_cord_y, start_cord_x: end_cord_x]
        roi_color = frame2[start_cord_y: end_cord_y, start_cord_x: end_cord_x]
        roi_scale = roi_color.shape[1] / empty_h
        roi_color_scaled = cv2.resize(roi_color, (int(empty_w), int(roi_color.shape[0] / roi_scale)))

        # print(base.shape)

        # save image on disk
        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        cv2.rectangle(frame2, (x, y), (x + w, y + h), blue_color, stroke)
        cv2.rectangle(frame2, (start_cord_x, start_cord_y), (end_cord_x, end_cord_y), red_color, stroke)

        print("base_shape")
        print(base.shape)
        print("roi_color")
        print(roi_color.shape)
        print("roi_color_scaled")
        print(roi_color_scaled.shape)

        w_2 = roi_color_scaled.shape[1] / 2
        h_2 = roi_color_scaled.shape[0] / 2

        if w_2 % 2 != 0:
            w_2 = w_2 - 1
        if h_2 % 2 != 0:
            h_2 = h_2 - 1

        print("w_2")
        print(w_2)
        print("h_2")
        print(h_2)

        roy_color_center_x = start_cord_x + roi_color.shape[1] / 2
        roy_color_center_y = start_cord_y + roi_color.shape[0] / 2
        cv2.circle(frame2, (roy_color_center_x, roy_color_center_y), 10, blue_color, -1)

        # print(roi_color.shape)
        # print(roy_color_center_x)
        # print(roy_color_center_y)
        # print(w_2)
        # print(h_2)

        tmp_x = empty_center_x - w_2
        tmp_y = empty_center_y - h_2

        if tmp_x % 2 != 0:
            tmp_x = tmp_x - 1
        if tmp_y % 2 != 0:
            tmp_y = tmp_y - 1

        print("tmp")
        print(tmp_x)
        print(tmp_y)

        base[tmp_y:tmp_y + roi_color_scaled.shape[0], tmp_x:tmp_x + roi_color_scaled.shape[1]] = roi_color_scaled
        # cv2.circle(background, (tmp_x, tmp_y), 10, red_color, -1)

    print("------------------")

    base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    background_levels = cv2.split(background)
    background_2_levels = cv2.split(background_2)
    # cv2.imshow("c0", c[0])

    background = cv2.merge((background_levels[0], background_levels[0], background_levels[0], background_2_levels[3]))

    alpha_s = background[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        base[0:bg_h, 0:bg_w, c] = (alpha_s * background[:, :, c] +
                                   alpha_l * base[0:bg_h, 0:bg_w, c])

    #output = overlay_transparent(base, background, 0, 0, (bg_w, bg_h))
    output = base

    alpha_s = red_level[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        output[0:bg_h, 0:bg_w, c] = (alpha_s * red_level[:, :, c] +
                                     alpha_l * output[0:bg_h, 0:bg_w, c])

    resized_output = cv2.resize(output, (int(output.shape[1] / 2), int(output.shape[0] / 2)))
    resized_frame = cv2.resize(frame2, (int(frame2.shape[1] / 2), int(frame2.shape[0] / 2)))

    grayOutput = cv2.cvtColor(resized_output, cv2.COLOR_BGRA2GRAY)

    cv2.imshow("resize_frame", resized_frame)
    cv2.imshow("resized_output", resized_output)
    cv2.imshow('grayOutput', grayOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
