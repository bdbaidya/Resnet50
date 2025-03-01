import cv2
import os
import argparse
import csv
from tqdm import tqdm

def extract_frames(video_path, annotation_path, output_dir, image_format='jpg', quality=95):
    """
    Extract frames from a video file based on annotation data
    and save them to organized directories
    
    Args:
        video_path (str): Path to input video file
        annotation_path (str): Path to annotation CSV file
        output_dir (str): Root output directory
        image_format (str): Output image format (default: jpg)
        quality (int): JPEG quality (0-100, default: 95)
    """
    # Create output directory structure
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    phase_dir = os.path.join(output_dir, video_name, "frames")
    os.makedirs(phase_dir, exist_ok=True)
    
    # Read annotation file
    annotated_frames = set()
    with open(annotation_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                frame_number = int(row[0])
                annotated_frames.add(frame_number)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_numbers = sorted(annotated_frames)
    
    pbar = tqdm(total=len(frame_numbers), desc=f"Processing {video_name}")
    
    # Create frame number to phase mapping
    frame_phase_map = {}
    with open(annotation_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                frame_number = int(row[0])
                phase = row[1].strip()
                frame_phase_map[frame_number] = phase

    # Create phase directories and save frames
    saved_count = 0
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        phase = frame_phase_map.get(frame_number, "Unknown")
        phase_clean = phase.replace(" ", "").replace("/", "_")
        
        # Create phase-specific directory
        phase_output_dir = os.path.join(output_dir, video_name, phase_clean)
        os.makedirs(phase_output_dir, exist_ok=True)
        
        # Save frame with consistent naming
        frame_filename = f"{frame_number}.{image_format}"
        output_path = os.path.join(phase_output_dir, frame_filename)
        
        if image_format.lower() in ['jpg', 'jpeg']:
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            cv2.imwrite(output_path, frame)
        
        saved_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    print(f"Extracted {saved_count} annotated frames from {video_name}")

def process_videos(video_dir, annotation_dir, output_dir, image_format='jpg', quality=95):
    """
    Process all videos with matching annotation files
    
    Args:
        video_dir (str): Directory containing video files
        annotation_dir (str): Directory containing annotation files
        output_dir (str): Root output directory
        image_format (str): Output image format
        quality (int): JPEG quality
    """
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) 
                 if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        annotation_file = os.path.join(annotation_dir, f"{video_name}.csv")
        
        if not os.path.exists(annotation_file):
            print(f"No annotation file found for {video_file}")
            continue
            
        video_path = os.path.join(video_dir, video_file)
        extract_frames(video_path, annotation_file, output_dir, image_format, quality)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotated Frame Extractor')
    parser.add_argument('-v', '--video_dir', type=str, required=True,
                      help='Directory containing video files')
    parser.add_argument('-a', '--annotation_dir', type=str, required=True,
                      help='Directory containing annotation files')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Root output directory for organized frames')
    parser.add_argument('-fmt', '--image_format', type=str, default='jpg',
                      choices=['jpg', 'png'], help='Output image format (default: jpg)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                      help='Image quality for JPEG (0-100, default: 95)')
    
    args = parser.parse_args()
    
    process_videos(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        image_format=args.image_format,
        quality=args.quality
    )