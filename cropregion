import cv2

clicked_points = []
clone = None

def click_event(event, x, y, flags, param):
    global clicked_points, clone

    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # add point
        clicked_points.append((x, y))
        # draw the point on image
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners", clone)

        print(f"Point {len(clicked_points)}: {x, y}")

def select_four_corners(image_path):
    global clicked_points, clone
    clicked_points = []

    img = cv2.imread("Image 2.png")
    if img is None:
        raise ValueError("Could not load image for corner selection")

    clone = img.copy()
    cv2.imshow("Select 4 corners", clone)
    cv2.setMouseCallback("Select 4 corners", click_event)

    print("INSTRUCTIONS:")
    print("Click the FOUR corners of the display in order:")
    print(" 1) Top-Left")
    print(" 2) Top-Right")
    print(" 3) Bottom-Right")
    print(" 4) Bottom-Left")
    print("Press 'q' when done.")

    # Wait until 4 points selected or user quits
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) >= 4:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    if len(clicked_points) != 4:
        raise ValueError("Error: You must click exactly 4 points.")

    print("Selected points:", clicked_points)
    return clicked_points