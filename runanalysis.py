#%%
%load_ext autoreload
%autoreload 2
#%%
import cropregion
import sevensegments
import length
import matplotlib.pyplot as plt

#%% Define file names
nm_video_path = "7.1.mp4"
muscle_video_path = "7.2.mp4"

#%% Find crop region for newtonmeter
nm_corners = cropregion.select_four_corners(nm_video_path)
print("Use these points in your crop:", nm_corners)

#%% Find seven segments
crop_points = nm_corners
segments = sevensegments.define_segments(nm_video_path, crop_points, 0.30)


#%% Analyse Newtonmeter, plot results
data = sevensegments.newtonmeter_analysis(nm_video_path, crop_points, segments=segments, start_frame=10, end_frame=11*30)

#%% Clean data? 
def filter_monotonic(data, jump_thresh=10):
    """
    Keeps only points consistent with:
    increasing -> flat -> decreasing
    """

    cleaned = []

    prev = None
    direction = 0  # 1 = up, 0 = flat, -1 = down

    for t, val in data:

        if val is None:
            cleaned.append((t, None))
            continue

        if prev is None:
            cleaned.append((t, val))
            prev = val
            continue

        diff = val - prev

        # classify movement
        if abs(diff) < 1e-3:
            new_dir = 0
        elif diff > 0:
            new_dir = 1
        else:
            new_dir = -1

        # first direction sets trend
        if direction == 0:
            direction = new_dir

        # enforce pattern consistency
        if direction == 1:  # increasing phase
            if diff < -jump_thresh:
                cleaned.append((t, None))
                continue

        elif direction == -1:  # decreasing phase
            if diff > jump_thresh:
                cleaned.append((t, None))
                continue

        # detect transition from up -> flat -> down
        if direction == 1 and new_dir == 0:
            direction = 0
        elif direction == 0 and new_dir == -1:
            direction = -1

        cleaned.append((t, val))
        prev = val

    return cleaned

cleaned_data = filter_monotonic(data)
times = [t for (t, v) in cleaned_data if v is not None]
values = [v/10 for (t, v) in cleaned_data if v is not None]
plt.scatter(times, values)
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.show()

#%% Find crop region for muscle
muscle_corners = cropregion.select_rectangle(muscle_video_path)
print("Use these points in your crop:", muscle_corners)

#%% Select muscle end
end_coord = length.select_muscle_end(muscle_video_path)
print("End coord:", end_coord)

#%% Analyse length
lengths = length.length_analysis(video_path=muscle_video_path, 
                                 crop_points=muscle_corners, 
                                 end_coord=end_coord, 
                                 threshold = 220,
                                 start_frame=10, 
                                 end_frame=11*30)

#%%
times = [t for (t, l1, l2, l3) in lengths if l2 is not None]
values = [l2 for (t, l1, l2, l3) in lengths if l2 is not None]
plt.scatter(times, values)
plt.ylabel('Length (pixels)')
plt.xlabel('Time (s)')
plt.show()


# %% Plot hysterysis loop!
dist = [
    l2
    for (t, l1, l2, l3), (t2, f) in zip(lengths, cleaned_data)
    if f is not None and l2 is not None
]

force = [
    f/10
    for (t, l1, l2, l3), (t2, f) in zip(lengths, cleaned_data)
    if f is not None and l2 is not None
]
plt.scatter(dist, force)
plt.ylabel("Force (N)")
plt.xlabel('Length (pixels)')


# %% Save results to csv
import numpy as np

data = np.column_stack((dist, force))

np.savetxt(
    "force_dist.csv",
    data,
    delimiter=",",
    header="dist,force",
    comments=""
)
# %%
