#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3D 시각화를 위한 필요한 라이브러리 임포트
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def load_3d_samples(file_path):
    """
    저장된 3D 샘플을 로드합니다.
    
    Args:
        file_path (str): 3D 샘플 파일 경로
        
    Returns:
        list: 3D 샘플 리스트
    """
    with open(file_path, "rb") as fb:
        samples_3d = pickle.load(fb)
    return samples_3d

def create_3d_visualization(samples_3d, sample_indices=None, save_path=None):
    """
    Create 3D visualizations for selected samples.
    
    Args:
        samples_3d (list): List of 3D samples
        sample_indices (list): List of sample indices to visualize
        save_path (str): Path to save the result
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if sample_indices is None:
        sample_indices = list(range(min(5, len(samples_3d))))
    
    num_samples = len(sample_indices)
    fig = plt.figure(figsize=(15, 4 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(samples_3d):
            print(f"Warning: Index {idx} is out of range. Skipping.")
            continue
            
        sample = samples_3d[idx]
        ax = fig.add_subplot(num_samples, 1, i+1, projection='3d')
        
        # Draw the path
        ax.plot(sample['x'], sample['y'], sample['z'], 'r-', linewidth=2)
        
        # Mark start and end points
        ax.scatter(sample['x'][0], sample['y'][0], sample['z'][0], 
                   c='g', marker='o', s=100, label='Start')
        ax.scatter(sample['x'][-1], sample['y'][-1], sample['z'][-1], 
                   c='b', marker='o', s=100, label='End')
        
        # Show all points (smaller)
        ax.scatter(sample['x'], sample['y'], sample['z'], 
                   c=np.arange(len(sample['x'])), cmap='viridis', 
                   s=30, alpha=0.5)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sample {idx}')
        ax.legend()
        
        # Adjust view
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to {save_path}")
    
    return fig

def create_3d_animation(sample, save_path=None, fps=15, duration=10, rotate=True, normalize_scale=False):
    """
    Create an animation for a specific 3D sample.
    
    Args:
        sample (dict): 3D sample to animate
        save_path (str): Path to save the result (with extension)
        fps (int): Frames per second
        duration (int): Animation duration in seconds
        rotate (bool): Whether to add rotation animation
        normalize_scale (bool): Whether to normalize axis scaling
        
    Returns:
        matplotlib.animation.FuncAnimation: The generated animation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine data ranges
    x_range = (min(sample['x']), max(sample['x']))
    y_range = (min(sample['y']), max(sample['y']))
    z_range = (min(sample['z']), max(sample['z']))
    
    # Calculate padding
    x_pad = (x_range[1] - x_range[0]) * 0.1
    y_pad = (y_range[1] - y_range[0]) * 0.1
    z_pad = (z_range[1] - z_range[0]) * 0.1
    
    # Handle the case where a dimension has zero range
    if x_pad == 0: x_pad = 0.1
    if y_pad == 0: y_pad = 0.1
    if z_pad == 0: z_pad = 0.1
    
    # Calculate axis limits with padding
    if normalize_scale:
        max_range = max(
            x_range[1] - x_range[0], 
            y_range[1] - y_range[0],
            z_range[1] - z_range[0]
        )
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        z_center = (z_range[0] + z_range[1]) / 2
        
        x_min, x_max = x_center - max_range/2 - x_pad, x_center + max_range/2 + x_pad
        y_min, y_max = y_center - max_range/2 - y_pad, y_center + max_range/2 + y_pad
        z_min, z_max = z_center - max_range/2 - z_pad, z_center + max_range/2 + z_pad
    else:
        x_min, x_max = x_range[0] - x_pad, x_range[1] + x_pad
        y_min, y_max = y_range[0] - y_pad, y_range[1] + y_pad
        z_min, z_max = z_range[0] - z_pad, z_range[1] + z_pad
    
    # Show full path in the background
    ax.plot(sample['x'], sample['y'], sample['z'], 'gray', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.scatter(sample['x'][0], sample['y'][0], sample['z'][0], 
               c='g', marker='o', s=100, label='Start')
    ax.scatter(sample['x'][-1], sample['y'][-1], sample['z'][-1], 
               c='b', marker='o', s=100, label='End')
    
    # Current position point
    point, = ax.plot([], [], [], 'ro', markersize=10)
    
    # Current path line
    line, = ax.plot([], [], [], 'r-', linewidth=2)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Path Animation')
    
    # Set view range
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Add equal aspect ratio if normalizing
    if normalize_scale:
        ax.set_box_aspect([1, 1, 1])
    
    # Initialize function
    def init():
        point.set_data([], [])
        point.set_3d_properties([])
        line.set_data([], [])
        line.set_3d_properties([])
        return point, line
    
    # Animation frame function
    def animate(i):
        # Path animation
        idx = min(i, len(sample['x'])-1)  # Prevent index from going out of bounds
        
        # Update current position
        point.set_data([sample['x'][idx]], [sample['y'][idx]])
        point.set_3d_properties([sample['z'][idx]])
        
        # Update path up to current position
        line.set_data(sample['x'][:idx+1], sample['y'][:idx+1])
        line.set_3d_properties(sample['z'][:idx+1])
        
        # Add rotation animation (optional)
        if rotate:
            ax.view_init(elev=30, azim=i)
        
        return point, line
    
    frames = max(len(sample['x']), 360 if rotate else 1)
    interval = duration * 1000 / frames  # Convert to milliseconds
    
    anim = FuncAnimation(fig, animate, frames=frames, 
                         init_func=init, interval=interval, 
                         blit=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    plt.close(fig)  # Free memory
    return anim

def create_multiple_views(sample, save_path=None, normalize_scale=False, figsize=(16, 12), dpi=300):
    """
    Create visualizations of a 3D sample from different viewpoints.
    
    Args:
        sample (dict): 3D sample to visualize
        save_path (str): Path to save the result
        normalize_scale (bool): Whether to normalize axis scaling for better visualization
        figsize (tuple): Figure size in inches (width, height)
        dpi (int): Resolution for saved image
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig = plt.figure(figsize=figsize)
    
    views = [
        (0, 0, "Front View"),
        (0, 90, "Side View"),
        (90, 0, "Top View"),
        (30, 45, "Perspective View")
    ]
    
    # Determine data ranges for consistent scaling across views
    x_range = (min(sample['x']), max(sample['x']))
    y_range = (min(sample['y']), max(sample['y']))
    z_range = (min(sample['z']), max(sample['z']))
    
    # Calculate padding for better visualization
    x_pad = (x_range[1] - x_range[0]) * 0.1
    y_pad = (y_range[1] - y_range[0]) * 0.1
    z_pad = (z_range[1] - z_range[0]) * 0.1
    
    # Handle the case where a dimension has zero range
    if x_pad == 0: x_pad = 0.1
    if y_pad == 0: y_pad = 0.1
    if z_pad == 0: z_pad = 0.1
    
    # Calculate limits with padding
    x_lim = (x_range[0] - x_pad, x_range[1] + x_pad)
    y_lim = (y_range[0] - y_pad, y_range[1] + y_pad)
    z_lim = (z_range[0] - z_pad, z_range[1] + z_pad)
    
    # Find the maximum range to normalize all axes to the same scale
    if normalize_scale:
        max_range = max(
            x_range[1] - x_range[0], 
            y_range[1] - y_range[0],
            z_range[1] - z_range[0]
        )
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        z_center = (z_range[0] + z_range[1]) / 2
        x_lim = (x_center - max_range/2 - x_pad, x_center + max_range/2 + x_pad)
        y_lim = (y_center - max_range/2 - y_pad, y_center + max_range/2 + y_pad)
        z_lim = (z_center - max_range/2 - z_pad, z_center + max_range/2 + z_pad)
    
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Draw the path
        ax.plot(sample['x'], sample['y'], sample['z'], 'r-', linewidth=2)
        
        # Mark start and end points
        ax.scatter(sample['x'][0], sample['y'][0], sample['z'][0], 
                   c='g', marker='o', s=100, label='Start')
        ax.scatter(sample['x'][-1], sample['y'][-1], sample['z'][-1], 
                   c='b', marker='o', s=100, label='End')
        
        # Color points by time order
        colors = np.arange(len(sample['x']))
        scatter = ax.scatter(sample['x'], sample['y'], sample['z'], 
                          c=colors, cmap='viridis', s=30, alpha=0.7)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Apply consistent scaling across all views
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        
        # Set view
        ax.view_init(elev=elev, azim=azim)
        
        # Add equal aspect ratio for better 3D perception
        if normalize_scale:
            ax.set_box_aspect([1, 1, 1])
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=fig.axes, orientation='vertical', pad=0.05)
    cbar.set_label('Time Sequence')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Multi-view image saved to {save_path}")
    
    return fig

def main(args):
    # Convert figsize argument to tuple of floats
    figsize = (float(args.figsize[0]), float(args.figsize[1])) if args.figsize else (16.0, 12.0)
    print(f"Using figure size: {figsize}")
    
    # Load 3D samples
    try:
        samples_3d = load_3d_samples(args.input_file)
        print(f"Loaded {len(samples_3d)} 3D samples.")
    except Exception as e:
        print(f"Error loading 3D samples: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create basic 3D visualization
    create_3d_visualization(
        samples_3d,
        sample_indices=args.sample_indices,
        save_path=os.path.join(args.output_dir, "3d_visualization.png")
    )
    
    # Create multi-view visualizations for selected samples
    if args.multi_view:
        for idx in args.sample_indices:
            if idx < len(samples_3d):
                create_multiple_views(
                    samples_3d[idx],
                    save_path=os.path.join(args.output_dir, f"multi_view_sample_{idx}.png"),
                    normalize_scale=args.normalize_scale,
                    figsize=figsize,
                    dpi=args.dpi
                )
    
    # Create animations
    if args.create_animation:
        for idx in args.sample_indices:
            if idx < len(samples_3d):
                create_3d_animation(
                    samples_3d[idx],
                    save_path=os.path.join(args.output_dir, f"animation_sample_{idx}.gif"),
                    fps=args.fps,
                    duration=args.duration,
                    rotate=args.rotate,
                    normalize_scale=args.normalize_scale
                )
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    # 입력 인자 파싱
    parser = argparse.ArgumentParser(description="TimeGAN 생성 3D 데이터 시각화")
    
    parser.add_argument(
        '--input_file',
        required=True,
        type=str,
        help='3D 샘플이 저장된 pickle 파일 경로'
    )
    
    parser.add_argument(
        '--output_dir',
        default='./3d_visualizations',
        type=str,
        help='시각화 결과 저장 디렉토리'
    )
    
    parser.add_argument(
        '--sample_indices',
        default=[0, 1, 2, 3, 4],
        nargs='+',
        type=int,
        help='시각화할 샘플 인덱스'
    )
    
    parser.add_argument(
        '--multi_view',
        action='store_true',
        help='Create multi-view visualizations'
    )
    
    parser.add_argument(
        '--create_animation',
        action='store_true',
        help='Create animations'
    )
    
    parser.add_argument(
        '--fps',
        default=15,
        type=int,
        help='Animation frames per second'
    )
    
    parser.add_argument(
        '--duration',
        default=10,
        type=int,
        help='Animation duration in seconds'
    )
    
    parser.add_argument(
        '--rotate',
        action='store_true',
        help='Add rotation effect to animations'
    )
    
    parser.add_argument(
        '--normalize_scale',
        action='store_true',
        help='Normalize axis scales for better visualization'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[16, 12],
        help='Figure size (width, height) in inches'
    )
    
    parser.add_argument(
        '--dpi',
        default=300,
        type=int,
        help='Resolution of saved images'
    )
    
    args = parser.parse_args()
    
    main(args) 