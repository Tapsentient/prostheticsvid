import cropregion
import numpy as np
from sklearn.cluster import KMeans
import cv2

def two_blob_kmeans(frame, previous_centroid=None, dist_thresh=30):
    '''
    Returns the positions of centroids of the aluminium foil in each frame. 
    Format: c1, c2 where ci = (xi, yi)
    '''

    # Get coordinates of all white pixels
    ys_white, xs_white = np.where(frame == 255)

    # If not enough white pixels → blank frame
    if len(xs_white) < 2:
        c1 = (np.nan, np.nan)
        c2 = (np.nan, np.nan)
        return None, None

    coords = np.column_stack((xs_white, ys_white))  # shape: (N, 2)

    if previous_centroid is not None:
        prev_c1, prev_c2 = previous_centroid
        d1 = np.linalg.norm(coords - prev_c1, axis=1)
        d2 = np.linalg.norm(coords - prev_c2, axis=1)
        mask = (d1 < dist_thresh) | (d2 < dist_thresh)
        coords = coords[mask]


    # If too few points for 2 clusters, KMeans fails → skip
    if coords.shape[0] < 2:
        c1 = (np.nan, np.nan)
        c2 = (np.nan, np.nan)
        return None, None

    # Run k-means to split into two clusters
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(coords)

    c = kmeans.cluster_centers_
    
    # c is shape (2, 2): [ [x1,y1], [x2,y2] ]
    c1 = (c[0][0], c[0][1])
    c2 = (c[1][0], c[1][1])

    return c1, c2

def length_analysis(video_path, crop_points=None, end_coord=(0, 0), threshold=125, dist_thresh = 20, start_frame=0, end_frame=None):
    '''
    Returns the length of the muscle in each frame. In order to do so, image converted to BnW to isolate foil
    crop_points must be of the format x, y, w, h. cropregion.select_rectangle can help select these.
    Return format: [(time, end_to_end_length, distance between foils, distance between mid foil and end)]
    thresh: Opacity below which pixel set to black. Tweak for noise
    dist_thresh: All points this distance away from the centroid in the previous frame set to black. Tweak for noise     
    '''
    cap = cv2.VideoCapture(video_path) #Open video file
    
    #Check if file opened or not
    if not cap.isOpened():
        print('Error: Could not open video')
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #Set start frame

    #If end_frame is not provided, set it to the last frame
    if end_frame is None: 
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Print video qualities
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video Properties \nFrame Width, Frame Height, FPS: ")
    print(frame_width, frame_height, fps)
    x, y, w, h = crop_points
    data = []
    previous_centroids = None
    while True: 
        ret, frame = cap.read()

        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
            break #Break if at last frame or if exceeded end frame

        if crop_points is not None:
            cropped_frame = frame[y:y+h, x:x+w]

        cv2.imshow('Original Frame', cropped_frame)
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0) #Blur out imperfections
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                
        cv2.imshow("Post image processing", thresh)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.waitKey(1)
            cv2.waitKey(1)
            break

        c1, c2 = two_blob_kmeans(thresh, previous_centroids, dist_thresh)
        mid_length_1 = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        end_length_1 = np.sqrt((c1[0]-end_coord[0])**2 + (c1[1]-end_coord[1])**2)
        end_length_2 = np.sqrt((c2[0]-end_coord[0])**2 + (c2[1]-end_coord[1])**2)
        if end_length_1<end_length_2:
            end_to_end_length = end_length_2
            mid_length_2 = end_length_1
        else: 
            end_to_end_length = end_length_2
            mid_length_2 = end_length_1     
        
        thresh_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.circle(thresh_vis, (int(c1[0]), int(c1[1])), 1, (0, 0, 255))
        cv2.circle(thresh_vis, (int(c2[0]), int(c2[1])), 1, (0, 0, 255))
        #Record time
        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        data.append((time_sec, end_to_end_length, mid_length_1, mid_length_2))
        previous_centroids = c1, c2
        cv2.circle(thresh_vis,
                (int(c1[0]), int(c1[1])),
                dist_thresh,
                (0, 255, 0), 2)

        cv2.circle(thresh_vis,
                (int(c2[0]), int(c2[1])),
                dist_thresh,
                (0, 0, 255), 2)
        
        cv2.imshow('With centroid, radius', thresh_vis)



    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return data

clicked_point = None
clone = None
def click_event(event, x, y, flags, param):
    global clicked_point, clone

    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # add point
        clicked_point = (x, y)
        # draw the point on image
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select end point", clone)


def select_muscle_end(video_path):
    global clicked_point, clone
    clicked_point = None

    img = cropregion.nth_frame(video_path, 0.5)
    if img is None:
        raise ValueError("Could not load image for corner selection")

    clone = img.copy()
    cv2.imshow("Select muscle end", clone)
    cv2.setMouseCallback("Select muscle end", click_event)

    # Wait until 4 points selected or user quits
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or clicked_point is not None:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


    print("Selected point:", clicked_point)
    return clicked_point

