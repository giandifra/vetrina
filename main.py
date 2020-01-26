import cv2
import numpy as np
import time


def overlay_with_opacity(bg, overlay, opacity):
    bg_copy = bg.copy()
    overaly_copy = overlay.copy()
    bg_h, bg_w, bg_c = bg_copy.shape
    base = np.zeros((bg_h, bg_w, bg_c), dtype='uint8')
    print(1 - opacity)
    cv2.addWeighted(base, opacity, overaly_copy, 1 - opacity, 0, base)
    alpha_s = base[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        bg_copy[0:bg_h, 0:bg_w, c] = (alpha_s * base[:, :, c] +
                                      alpha_l * bg_copy[0:bg_h, 0:bg_w, c])
    return bg_copy


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
red_level = cv2.imread("memoria-3-rosso-nuovo.png", cv2.IMREAD_UNCHANGED)
bg_h, bg_w, bg_c = background.shape
print(background.shape)

# posizione in cui inserire il crop del viso
# x_offset = 1786
# y_offset = 650

x_offset = 562
y_offset = 160

# empty_width = 84
# empty_height = 100
# empty_center_x = x_offset + empty_width / 2
# empty_center_y = y_offset + empty_height / 2

empty_h = 138
empty_w = 100
empty_aspectRatio = empty_w / empty_h
empty_center_x = 678
empty_center_y = 195
print(empty_center_x)
print(empty_center_y)

padding = 30

# B G R
blue_color = (255, 0, 0)
red_color = (0, 0, 255)
stroke = 2

cap = cv2.VideoCapture()
cap.open(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
# cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
focus = 10
cap.set(28, focus)

print("VideoCapture open with frame ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
global start
start = None

while True:

    # 1280x720 (camera mac GM)
    ret, frame = cap.read()
    print(frame.shape)
    # base nera con 4 canali BGRA
    x = int(frame.shape[1] / 3)
    black_base = np.zeros(frame.shape, frame.dtype)
    roi = frame[0:frame.shape[0], x:x + x]
    black_base[0:frame.shape[0], x:x + x] = roi
    frame = black_base

    scale = background.shape[1] / frame.shape[1]
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_resized = cv2.resize(frame, (background.shape[1], int(background.shape[0] * scale)))

    grayFrame2 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame2, scaleFactor=1.5, minNeighbors=5, minSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    min_x = int(frame_resized.shape[1] / 3)
    max_x = min_x * 2
    print(min_x)
    print(max_x)

    array = []
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        base = np.zeros((bg_h, bg_w, bg_c), dtype='uint8')
        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        start_cord_x = x
        start_cord_y = y - padding
        end_cord_x = x + w
        new_h = h + padding * 2
        end_cord_y = start_cord_y + new_h
        roi_gray = grayFrame2[start_cord_y:end_cord_y, start_cord_x:end_cord_x]  # gray
        # roi_color = frame2[start_cord_y: end_cord_y, start_cord_x: end_cord_x]
        roi_scale = roi_gray.shape[0] / empty_h
        roi_aspectRatio = roi_gray.shape[1] / roi_gray.shape[0]
        print("ROI GRAY SHAPE ", roi_gray.shape, "px, aspect ratio ", roi_aspectRatio, " fitting into ratio ",
              empty_aspectRatio)
        if roi_aspectRatio <= empty_aspectRatio:
            # Aspect ratio di viso rilevato più "verticale" di destinazione, fissa larghezza
            dest_w = int(empty_w)
            dest_h = int(empty_h / roi_aspectRatio)
        else:
            # Aspect ratio di viso rilevato più "orizzontale" di destinazione, fissa altezza
            dest_w = int(empty_w / roi_aspectRatio)
            dest_h = int(empty_h)
        print("Destination size ", dest_w, ",", dest_h, "px")
        roi_gray_scaled = cv2.resize(roi_gray, (dest_w, dest_h))
        # print(base.shape)

        # save image on disk
        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), blue_color, stroke)
        cv2.rectangle(frame_resized, (start_cord_x, start_cord_y), (end_cord_x, end_cord_y), red_color, stroke)

        roy_color_center_x = int(start_cord_x + roi_gray.shape[1] / 2)
        roy_color_center_y = int(start_cord_y + roi_gray.shape[0] / 2)
        cv2.circle(frame_resized, (roy_color_center_x, roy_color_center_y), 10, blue_color, -1)

        roi_gray_bgra = cv2.cvtColor(roi_gray_scaled, cv2.COLOR_GRAY2BGRA)
        array.append(roi_gray_bgra)
        # cv2.circle(background, (tmp_x, tmp_y), 10, red_color, -1)

    print("------------------")
    if len(array) == 0:
        print("empty array")
        if start is None or time.time() - start > 4:
            print("maggio di 3 o nullo")
            final_base = cv2.resize(background_base,
                                    (int(background_base.shape[1] / 2), int(background_base.shape[0] / 2)))
            cv2.imshow("final", final_base)
        else:
            diff = time.time() - start
            if diff > 1.5:
                diff = diff - 1.5 + 0.375
                opacity = diff / 2.5
                print("opacity")
                print(opacity)
                if opacity <= 1:
                    if base.shape[2] == 3:
                        base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
                    final_base = overlay_with_opacity(base, red_level, opacity)
                    final_base = cv2.resize(final_base,
                                            (int(final_base.shape[1] / 2), int(final_base.shape[0] / 2)))
                    cv2.imshow("final", final_base)
    else:
        print(len(array))
        start = time.time()
        roi_gray_bgra = array[0]
        cv2.imshow("roi_gray_bgra", roi_gray_bgra)

        # Aumento la luminosita
        value = 150
        roi_gray_bgra = np.where((255 - roi_gray_bgra) < value, 255, roi_gray_bgra + value)
        cv2.imshow("grey_new", roi_gray_bgra)
        # print("base_shape")
        # print(base.shape)
        # print("roi_gray")
        # print(roi_gray_scaled.shape)
        # print("roi_gray_scaled")
        # print(roi_gray_scaled.shape)

        w_2 = int(roi_gray_bgra.shape[1] / 2)
        h_2 = int(roi_gray_bgra.shape[0] / 2)

        if w_2 % 2 != 0:
            w_2 = w_2 - 1
        if h_2 % 2 != 0:
            h_2 = h_2 - 1

        print("w_2")
        print(w_2)
        print("h_2")
        print(h_2)

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
        base[tmp_y:tmp_y + roi_gray_bgra.shape[0], tmp_x:tmp_x + roi_gray_bgra.shape[1]] = roi_gray_bgra
        base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
        background_levels = cv2.split(background)
        background_2_levels = cv2.split(background_2)
        # cv2.imshow("c0", c[0])

        # Faccio il merge dei livelli per otteren un immagine BGRA
        background = cv2.merge(
            (background_levels[0], background_levels[0], background_levels[0], background_2_levels[3]))

        # Inserisco il background sopra il livello base con il crop del viso
        alpha_s = background[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            base[0:bg_h, 0:bg_w, c] = (alpha_s * background[:, :, c] +
                                       alpha_l * base[0:bg_h, 0:bg_w, c])

        # output = overlay_transparent(base, background, 0, 0, (bg_w, bg_h))
        output = base

        # Inserisco il mantello rosso sopra l'immagine comprendente il viso estratto
        alpha_s = red_level[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            output[0:bg_h, 0:bg_w, c] = (alpha_s * red_level[:, :, c] +
                                         alpha_l * output[0:bg_h, 0:bg_w, c])

        # Resize
        resized_output = cv2.resize(output, (int(output.shape[1] / 2), int(output.shape[0] / 2)))
        cv2.imshow("final", resized_output)

        # grayOutput = cv2.cvtColor(resized_output, cv2.COLOR_BGRA2GRAY)
        # cv2.imshow('grayOutput', grayOutput)

    frame_resized = cv2.resize(frame_resized, (int(frame_resized.shape[1] / 2), int(frame_resized.shape[0] / 2)))
    print("frame_resized")
    print(frame_resized.shape)
    cv2.imshow("resize_frame", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
