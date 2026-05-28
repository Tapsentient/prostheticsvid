#%%
%load_ext autoreload
%autoreload 2
#%%
import cropregion
import sevensegments
import length
import matplotlib.pyplot as plt

#%% Define file names
nm_video_path = "NewtonmeterTest2.mp4"
muscle_video_path = "MuscleTest2.mp4"

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

#%% Select muscle end
end_coord = length.select_muscle_end(muscle_video_path)
print("End coord:", end_coord)

#%% Analyse length
lengths = length.length_analysis(video_path=muscle_video_path, 
                                 crop_points=muscle_corners, 
                                 end_coord=end_coord, 
                                 threshold = 220)

#%%
times = [t for (t, l1, l2, l3) in lengths if l2 is not None]
values = [l2 for (t, l1, l2, l3) in lengths if l2 is not None]
plt.scatter(times, values)
plt.ylabel('Length (pixels)')
plt.xlabel('Time (s)')
plt.show()


# %% Plot hysterysis loop!
dist = [l2 for t, l1, l2, l3 in lengths if l2 is not None]
force = [f for (t, f) in data if t is not None]
plt.scatter(dist, force)
plt.ylabel("Length (pixels)")
plt.xlabel('Time (s)')

