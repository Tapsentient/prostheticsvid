'''
Test
'''
#%%
%load_ext autoreload
%autoreload 2
#%%
import cropregion
import sevensegments
import length
import matplotlib.pyplot as plt

#%% Define file names
nm_video_path = "Newtonmeter2.mp4"
muscle_video_path = "Muscle.mp4"

#%% Find crop region for newtonmeter
nm_corners = cropregion.select_four_corners(nm_video_path)
print("Use these points in your crop:", nm_corners)

#%% Find seven segments
crop_points = nm_corners
segments = sevensegments.define_segments(nm_video_path, crop_points)


#%% Analyse Newtonmeter, plot results
data = sevensegments.newtonmeter_analysis(nm_video_path, crop_points, segments=segments)

times = [t for (t, v) in data if v is not None]
values = [v/10 for (t, v) in data if v is not None]
plt.scatter(times, values)
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.show()

#%% Find crop region for muscle
muscle_corners = cropregion.select_rectangle(muscle_video_path)
print("Use these points in your crop:", muscle_corners)

#%% Analyse length
lengths = length.length_analysis(muscle_video_path, muscle_corners)

times = [t for (t, v) in lengths if v is not None]
values = [v/10 for (t, v) in lengths if v is not None]
plt.scatter(times, values)
plt.ylabel('Length (pixels)')
plt.xlabel('Time (s)')
plt.show()


# %%
