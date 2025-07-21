# Import Requried Libraries - Needs to be installed on the environment if not installed already
import pandas as pd
import os
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import pprint
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2

# Set the working directory
os.chdir("working_directory/")

# Directories for SMS text, Yield editor files and Output files
text_files_dir = "AgLeader Txt Files/"
rmcode_files_dir = "Yield Editor CSV files/"
output_dir = "Outputs"

# Define the header row as a list of column names (AgLeader SMS files do not come with the column name)
header = ["Longitude", "Latitude", "Grain Flow", "GPS Time", "Logging Interval", "Distance", "Swath", "Moisture", "Header Status", "Pass", "Serial Number", "Field ID", "Load ID", "Grain Type", "GPS Status", "PDOP", "Altitude", "RmCode"]

# Get a list of all text files in the directory
text_files = [f for f in os.listdir(text_files_dir) if f.endswith('.txt')]


# Function to compute valid min/max yield on a field
def compute_min_max_yield(yield_data, YUlim, YLlim, Yscale, MINYabs):
    Q_YUlim = np.quantile(yield_data, YUlim)
    Q_YLlim = np.quantile(yield_data, YLlim)
    X = Q_YUlim - Q_YLlim
    MINY = Q_YLlim - (Yscale * X)
    MINY = max(MINY, MINYabs)
    MAXY = Q_YUlim + (Yscale * X)
    return MINY, MAXY

# Function to compute valid min/max velocity on a field
def compute_min_max_velocity(velocity_data, VUlim, VLlim, Vscale, MINVabs):
    Q_VUlim = np.quantile(velocity_data, VUlim)
    Q_VLlim = np.quantile(velocity_data, VLlim)
    X = Q_VUlim - Q_VLlim
    MINV = Q_VLlim - (Vscale * X)
    MINV = max(MINV, MINVabs)
    MAXV = Q_VUlim + (Vscale * X)
    return MINV, MAXV

# Threshold Parameters for yield and velocity (Followed Yield Editor's Method)
YUlim = 0.99
YLlim = 0.1
Yscale = 0.25
MINYabs = 1

VUlim = 0.99
VLlim = 0.08
Vscale = 0.18
MINVabs = 0.5


########### Start - Functions for spatial features of the field #############

# Function to get field boundary (Convex Hull)
def get_field_boundary(df):
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    return points.unary_union.convex_hull

# Compute geometric moments
def compute_moments(polygon, max_order=4):
    coords = np.array(polygon.exterior.coords)
    x = coords[:, 0]  # X-coordinates
    y = coords[:, 1]  # Y-coordinates
    
    area = polygon.area
    centroid_x = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)
    centroid_y = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * area)

    # Compute central moments relative to centroid
    x_c = x - centroid_x
    y_c = y - centroid_y
    
    moments = {(0, 0): area}

    for i in range(max_order + 1):
        for j in range(max_order + 1 - i):
            moments[(i, j)] = np.sum(x_c**i * y_c**j)
    
    return moments

# Compute triangularity using moment matching method (Rosin, 2003)
def compute_triangularity(polygon):
    polygon_moments = compute_moments(polygon)
    
    # Prototype equilateral triangle centered at the origin
    prototype_triangle = Polygon([(-0.5, -0.288), (0.5, -0.288), (0, 0.577)])
    triangle_moments = compute_moments(prototype_triangle)

    # Compute triangularity index
    moment_diff_sum = sum((triangle_moments.get((i, j), 0) - polygon_moments.get((i, j), 0)) ** 2 
                          for i in range(5) for j in range(5 - i))
    
    TM = 1 / (1 + moment_diff_sum)
    return TM

# Compute ellipticity using moment invariant method (Rosin, 2003)
def compute_ellipticity(polygon):
    moments = compute_moments(polygon)
    l20 = moments.get((2, 0), 0)
    l02 = moments.get((0, 2), 0)
    l11 = moments.get((1, 1), 0)
    l00 = moments.get((0, 0), 1)  # ✅ Fix: l00 is polygon area

    I1 = (l20 * l02 - l11**2) / (l00**4)

    # Compute ellipticity index
    if I1 <= 1:
        EI = 16 * np.pi**2 * I1
    else:
        EI = 16 * np.pi**2 / I1

    return EI

# Compute shape indices
def compute_shape_indices(polygon):
    area = polygon.area
    perimeter = polygon.length
    
    # Compactness
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    
    # Rectangularity (Using Rotating Caliper MBR)
    min_rect = polygon.minimum_rotated_rectangle
    rectangularity = area / min_rect.area

    # Ellipticity using moment invariant method (Rosin 2003)
    ellipticity = compute_ellipticity(polygon)

    # Triangularity using moment matching method (Rosin 2003)
    triangularity = compute_triangularity(polygon)

    return {
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Ellipticity": ellipticity,
        "Triangularity": triangularity
    }
########### End - Functions for spatial features of the field#############


########### Start - Functions for edge data points#############
# Function to calculate heading between two points
def calculate_heading(lat1, lon1, lat2, lon2):
    delta_lon = lon2 - lon1
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(delta_lon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(delta_lon))
    heading = np.degrees(np.arctan2(x, y))
    heading = (heading + 360) % 360
    return heading

# Function to adjust for quadrant changes
def adjust_direction_change(dir1, dir2):
    change = dir2 - dir1
    if change > 180:
        change -= 360
    elif change < -180:
        change += 360
    return change

# Function to calculate the cumulative distance from the start point
def cumulative_distance(points):
    distances = [0]
    for i in range(1, len(points)):
        # Access both latitude and longitude for the second point in each geodesic calculation
        dist = geodesic((points[i-1][0], points[i-1][1]), (points[i][0], points[i][1])).meters
        distances.append(distances[-1] + dist)
    return distances

threshold = 10
########### End - Functions for edge data points#############

######### Start - Function for detecting distance from the centroid ###########
# Function to calculate the Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon1 - lon2)
    
    a = sin(delta_phi / 2.0)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c
######### End - Function for detecting distance from the centroid ###########

######### Start - Processing files ###########

# List to store DataFrames
dataframes = []

# Run preprocessing for all the AgLeader SMS text files
for text_file in text_files:
    # Corresponding RmCode file
    rmcode_file = text_file.replace('.txt', '.csv')
    rmcode_path = os.path.join(rmcode_files_dir, rmcode_file)

    if not os.path.exists(rmcode_path):
        print(f"RmCode file for {text_file} not found. Skipping this file.")
        continue# Read and process the text file
    raw_df = pd.read_csv(os.path.join(text_files_dir, text_file), delimiter=',',  header=None)
    raw_df.columns = ["Longitude", "Latitude", "Grain_Flow", "GPS_Time", "Logging_Interval", "Distance", "Swath", "Moisture", "Header_Status", "Pass", "Serial_Number", "Field_ID", "Load_ID", "Grain_Type", "GPS_Status", "PDOP", "Altitude"]

    rmcode_df = pd.read_csv(rmcode_path, header=None)
    rmcode_df.columns = ["UTM_Easting", "UTM_Northing", "Latitude", "Longitude", "Yield", "Moisture", "Swath_Width", "Trvl_Dstnc", "Grain_Flow", "Interval", "Trnsct_Nmbr", "GPS_Time", "UTM_Zone", "RmCode"]
    
    raw_df['Latitude'] = raw_df['Latitude'].astype(float)
    raw_df['Longitude'] = raw_df['Longitude'].astype(float)
    rmcode_df['Latitude'] = rmcode_df['Latitude'].astype(float)
    rmcode_df['Longitude'] = rmcode_df['Longitude'].astype(float)

    raw_df_cleaned = raw_df.drop_duplicates(subset=['Latitude', 'Longitude'])
    rmcode_cleaned = rmcode_df.drop_duplicates(subset=['Latitude', 'Longitude'])
    merged_df = pd.merge(raw_df_cleaned, rmcode_cleaned[['Latitude', 'Longitude', 'RmCode']], on=['Latitude', 'Longitude'], how='left')

    df = merged_df[merged_df['RmCode'].notna()].copy()
    df['RmCode'] = df['RmCode'].apply(lambda x: 1 if x > 0 else 0)
    df['Velocity'] = (df['Distance'] * 3600) / (df['Logging_Interval'] * 63360)
    
    # Map crop density and market moisture by grain type
    df['Crop_Density'] = df['Grain_Type'].str.upper().map({'SOYBEANS': 60, 'CORN': 56})
    df['Market_Moisture'] = df['Grain_Type'].str.upper().map({'SOYBEANS': 0.13, 'CORN': 0.155})
    
    # Now calculate yield for each row
    df['Yield'] = (
        df['Grain_Flow'] * df['Logging_Interval'] * 43560 * (1 - (df['Moisture'] / 100))
    ) / (
        df['Distance'] * df['Swath'] * df['Crop_Density'] * (1 - df['Market_Moisture'])
    )
    
    # Replace infinite values with NaN only for the 'Yield' and 'Velocity' columns
    df['Yield'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Velocity'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Replace NaN values with the mean only for the 'Yield' and 'Velocity' columns
    df['Yield'].fillna(df['Yield'].mean(), inplace=True)
    df['Velocity'].fillna(df['Velocity'].mean(), inplace=True)
    
    #Add velocity variation
    df['Vel_Var'] = df['Velocity'].pct_change().abs()
    # Replace NaN values with the mean only for the 'velocity_ratio' column
    df['Vel_Var'].fillna(df['Vel_Var'].mean(), inplace=True)
    df.loc[df['Vel_Var'] > 0.2, 'RmCode'] = 1
    
    # Compute valid yield and velocity threshold
    MINY, MAXY = compute_min_max_yield(df['Yield'], YUlim, YLlim, Yscale, MINYabs)
    MINV, MAXV = compute_min_max_velocity(df['Velocity'], VUlim, VLlim, Vscale, MINVabs)
    
    df['MINY'] = MINY
    df['MAXY'] = MAXY
    df['MINV'] = MINV
    df['MAXV'] = MAXV
    
    #######Start - Shape Indices########
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

    # Create Field Boundary (Convex Hull)
    field_boundary = gdf.unary_union.convex_hull

    # Plot the field with yield data
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot for yield data points
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["Grain_Flow"], cmap="viridis", s=10, alpha=0.7)

    # Plot field boundary
    gpd.GeoSeries(field_boundary).plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2)

    # Colorbar for yield values
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Yield")

    # Labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Field Shape and Yield Distribution")

    # Show plot
    plt.show()
    
    # Compute indices for the field
    polygon = get_field_boundary(df)
    shape_indices = compute_shape_indices(polygon)

    #Display Result
    pprint.pprint(shape_indices)
    
    #populate the results into the dataset
    df["Compactness"] = shape_indices["Compactness"]
    df["Rectangularity"] = shape_indices["Rectangularity"]
    df["Ellipticity"] = shape_indices["Ellipticity"]
    df["Triangularity"] = shape_indices["Triangularity"]
    #######End - Shape Indices########
    
    #######Start - Edge data calculation########
    # Calculate heading for each measurement
    df['heading'] = df.apply(
        lambda row: calculate_heading(
            row['Latitude'], row['Longitude'],
            df.loc[row.name + 1, 'Latitude'] if (row.name + 1) in df.index else row['Latitude'],
            df.loc[row.name + 1, 'Longitude'] if (row.name + 1) in df.index else row['Longitude']
        ),
        axis=1
    )
    
    df['prev_avg_direction'] = df['heading'].rolling(window=5, min_periods=1).mean().shift(1)
    df['fwd_avg_direction'] = df['heading'].rolling(window=5, min_periods=1).mean().shift(-1)
    df['direction_change'] = df.apply(lambda row: adjust_direction_change(row['prev_avg_direction'], row['heading']), axis=1)
    df['future_direction_change'] = df.apply(lambda row: adjust_direction_change(row['heading'], row['fwd_avg_direction']), axis=1)

    df['start_turn'] = (abs(df['direction_change']) > threshold) & (abs(df['future_direction_change']) > threshold)
    df['end_turn'] = (abs(df['direction_change']) > threshold) & (abs(df['future_direction_change']) <= threshold)

    df['row_start'] = False
    df['row_end'] = False
    df.loc[0, 'row_start'] = True
    turn_active = False
    row_start_index = None
    df['near_edge'] = False
    
    for row in df.itertuples():
        if row.start_turn:
            df.loc[row.Index, 'row_end'] = True
            turn_active = True
        if turn_active and not row.start_turn:
            df.loc[row.Index, 'row_start'] = True
            turn_active = False

    for row in df.itertuples():
        if row.row_start:
            row_start_index = row.Index
            row_points = []
            turn_active = False
        if row_start_index is not None:
            row_points.append((row.Latitude, row.Longitude))
            if row.row_end or row.Index == len(df) - 1:
                distances = cumulative_distance(row_points)
                for i, dist in enumerate(distances):
                    if dist <= 5 or distances[-1] - dist <= 5:
                        df.loc[row_start_index + i, 'near_edge'] = True
                turn_active = True
                row_start_index = None
        if turn_active and not row.row_start:
            df.loc[row.Index, 'near_edge'] = True
        if row.start_turn or row.end_turn:
            df.loc[row.Index, 'near_edge'] = True
            
    df['near_edge'] = df['near_edge'].apply(lambda x: 'near_edge' if x else 'not_near_edge')

    edge_points = df[df['near_edge'] == 'near_edge']
    total_edge_points = len(edge_points)
    erroneous_edge_points = edge_points['RmCode'].sum()
    percentage_erroneous_edge_points = (erroneous_edge_points / total_edge_points) * 100 if total_edge_points > 0 else 0
        
    print(f"{text_file}: {percentage_erroneous_edge_points:.2f}% erroneous edge points")
    #######End - Edge data calculation########
    
    #######Start - Distance from centroid########
    # Calculate the centroid
    centroid_lat = df['Latitude'].mean()
    centroid_long = df['Longitude'].mean()
    
    # Calculate the distance from the centroid for each point
    df['Dist_Centroid'] = df.apply(
        lambda row: haversine(row['Latitude'], row['Longitude'], centroid_lat, centroid_long), axis=1
    )

    #######End - Distance from centroid########
    dataframes.append(df)
    print(f"Processed {text_file}")
    
######### End - Processign files ###########

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.drop(columns=['heading', 'prev_avg_direction', 'fwd_avg_direction', 
                 'direction_change', 'future_direction_change', 'start_turn', 
                 'end_turn', 'row_start', 'row_end', 'Longitude', 'Latitude', 
                 'GPS_Time', 'Serial_Number', 'Field_ID', 'Load_ID', 
                 'Grain_Type', 'GPS_Status', 'PDOP', 'Pass', 'Header_Status', 'Crop_Density', 'Market_Moisture'], inplace=True)

# Save the combined DataFrame to a CSV file
combined_csv_file = os.path.join(output_dir, "combined_data.csv")
combined_df.to_csv(combined_csv_file, index=False)
print(f"Combined data saved as {combined_csv_file}")