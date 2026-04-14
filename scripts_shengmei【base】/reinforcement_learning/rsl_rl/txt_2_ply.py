import numpy as np
import os

def convert_txt_to_ply():
    """Simple script to convert your specific file"""
    
    # Your file path
    input_file = "/home/roborock/R50_pc/grab.txt"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        print("Please check the path and try again.")
        return
    
    # Create output file path (same name with .ply extension)
    output_file = os.path.splitext(input_file)[0] + ".ply"
    
    try:
        # Load the data
        print(f"Loading data from: {input_file}")
        data = np.loadtxt(input_file)
        
        # Get number of points
        num_points = len(data)
        print(f"Found {num_points} points")
        
        # Create PLY file
        print(f"Creating PLY file: {output_file}")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")
            
            # Write data
            for point in data:
                x, y, z, intensity = point
                f.write(f"{x} {y} {z} {intensity}\n")
        
        print(f"Successfully created {output_file}")
        
        # Show some statistics
        print("\nData Statistics:")
        print(f"X range: {data[:, 0].min():.6f} to {data[:, 0].max():.6f}")
        print(f"Y range: {data[:, 1].min():.6f} to {data[:, 1].max():.6f}")
        print(f"Z range: {data[:, 2].min():.6f} to {data[:, 2].max():.6f}")
        print(f"Intensity range: {data[:, 3].min():.6f} to {data[:, 3].max():.6f}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Run the conversion
convert_txt_to_ply()