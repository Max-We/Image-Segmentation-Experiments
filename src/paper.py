import cv2
import numpy as np


def detect_paper_cv2(img):
    """
    Finds the corners of a paper, marks its outline with a polygon and rectifies the perspective
    Based on https://stackoverflow.com/a/60941676
    """
    # 1. Detect paper corners
    corners, thresh_img, morph_img = find_paper_corners(img)
    cv2.imwrite("../output/paper_thresh.jpg", thresh_img)
    cv2.imwrite("../output/paper_morph.jpg", morph_img)

    # 2. Mark corners visually
    marked_paper_img = mark_paper_edges(corners, img)
    cv2.imwrite("../output/paper_marked.jpg", marked_paper_img)

    # 3. Un-skew img perspective
    rectified_img = rectify_paper_perspective(corners, img)
    output_file = "../output/paper.jpg"
    cv2.imwrite(output_file, rectified_img)
    print(f"Saved result to {output_file}")


def rectify_paper_perspective(corners, img):
    # Reformat input corners to x,y list
    # https://www.perplexity.ai/search/495b46c3-3798-4db3-adc8-4dc9bc5c6623?s=c
    in_corners = corners[:, 0, :].astype(np.float32)

    # Find out where the width and height is of the paper is, to avoid wrong rotation in the output
    # Because all corners are saved counterclockwise we only have to find out if the first line is horizontal / vertical
    first_line_slope = (in_corners[1][1] - in_corners[0][1]) / (in_corners[1][0] - in_corners[0][0])
    first_line_is_vertical = first_line_slope > 0.5 or first_line_slope < -0.5

    # Height / width is calculated from how big the actual paper is (not the entire image)
    # np.linalg.norm: https://www.perplexity.ai/search/495b46c3-3798-4db3-adc8-4dc9bc5c6623?s=c
    side_a_1 = np.linalg.norm(in_corners[0] - in_corners[1])
    side_a_2 = np.linalg.norm(in_corners[2] - in_corners[3])
    side_b_1 = np.linalg.norm(in_corners[1] - in_corners[2])
    side_b_2 = np.linalg.norm(in_corners[3] - in_corners[0])
    side_a_avg = round((side_a_1 + side_a_2) / 2)
    side_b_avg = round((side_b_1 + side_b_2) / 2)

    # vertical lines -> height
    # horizontal lines -> width
    if first_line_is_vertical:
        height, width = side_a_avg, side_b_avg
    else:
        height, width = side_b_avg, side_a_avg

    # This is the order opencv expects the coordinates in, otherwise the picture will be rotated / mirrored
    # Starting at 0,0 going counterclockwise
    out_corners = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    # The input corners which we generated in `find_paper_corners` are saved counterclockwise
    # We still have to find the element closest to (0,0), so that the input corners match our output corners
    # With this knowledge we can re-order the corners accordingly
    distances = np.apply_along_axis(np.linalg.norm, 1, in_corners)
    i_closest_to_zero = np.argmin(distances)
    in_corners = np.concatenate((in_corners[i_closest_to_zero:], in_corners[:i_closest_to_zero]))

    # Apply perspective transformation
    perspective_matrix = cv2.getPerspectiveTransform(in_corners, out_corners)
    rectified_img = cv2.warpPerspective(img, perspective_matrix, (width, height))

    return rectified_img


def mark_paper_edges(corners, img):
    """
    Marks the outline of the paper by connecting the corners as a polygon
    """
    marked_paper_img = img.copy()
    cv2.polylines(marked_paper_img, [corners], True, (0, 0, 255), 2, cv2.LINE_AA)

    return marked_paper_img


def find_paper_corners(img):
    """
    Detects the corners of a paper
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # do otsu threshold on gray image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    # draw white filled largest contour on black just as a check to see it got the correct region
    page = np.zeros_like(img)
    cv2.drawContours(page, [big_contour], 0, (255, 255, 255), -1)

    # get perimeter and approximate a polygon
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

    return corners, thresh, morph
