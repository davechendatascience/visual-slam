import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--normalize", action="store_true", help="Center and scale points for viewing")
    parser.add_argument("--max_points", type=int, default=500000, help="Maximum number of points to display")
    parser.add_argument("--backend", default="matplotlib", choices=["matplotlib"], help="Viewer backend (default: matplotlib)")
    parser.add_argument("--clean", action="store_true", help="Hide axes and background for a clean look")
    args = parser.parse_args()

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Dark background style
    plt.style.use('dark_background')

    points = []
    colors = []
    print(f"Reading {args.ply}...")
    with open(args.ply, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x, y, z = map(float, parts[:3])
            r, g, b = map(int, parts[3:6])
            points.append([x, y, z])
            # Matplotlib expects RGB floats in [0, 1]
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    if not points:
        raise RuntimeError("PLY file is empty or could not be read.")
    
    print(f"Loaded {len(points)} points.")

    pts = np.array(points)
    cols = np.array(colors)

    # Keep original data for resampling
    original_pts = pts
    original_cols = cols

    # Initial global subsample
    if len(pts) > args.max_points:
        print(f"Subsampling to {args.max_points} points...")
        indices = np.random.choice(len(pts), args.max_points, replace=False)
        pts = pts[indices]
        cols = cols[indices]

    if args.normalize and len(original_pts) > 0:
        center = original_pts.mean(axis=0)
        original_pts = original_pts - center
        # Apply same transform to visible points
        pts = pts - center
        
        scale = np.max(np.linalg.norm(original_pts, axis=1))
        if scale > 0:
            original_pts = original_pts / scale
            pts = pts / scale

    print("Opening Matplotlib viewer...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store the scatter object to update it later
    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.5, alpha=0.6, depthshade=True)
    
    # Enforce equal physical proportions
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
    mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
    mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass

    if args.clean:
        ax.set_axis_off()
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        title = plt.title(f"Points: {len(pts)} (Press 'r' to refine)", color='white')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = plt.title(f"PLY Viewer ({len(pts)} points) - Press 'r' to refine")

    def refine_view(event=None):
        print("Refining view... calculating visible points...")
        # Get current projection matrix (incorporates rotation, zoom/dist, etc.)
        M = ax.get_proj()
        
        # Convert points to homogeneous coordinates (N, 4)
        # Optimization: operate in chunks if memory is tight, but 6M is OK for modern RAM.
        ones = np.ones((len(original_pts), 1))
        pts_hom = np.hstack((original_pts, ones))
        
        # Project to clip space: Res = P_hom @ M.T
        proj = pts_hom @ M.T
        
        # Perspective divide to get Normalized Device Coordinates (NDC)
        w = proj[:, 3:4]
        # Avoid division by zero warnings (though w=0 shouldn't happen for valid points in front of cam)
        mask_w = np.abs(w) > 1e-9
        
        # We only care about points where w != 0
        w[~mask_w] = 1.0 # arbitrary to avoid NaN
        
        ndc = proj[:, :3] / w
        
        # Matplotlib NDC limits are typically [-1, 1] for visible X and Y
        # We check this range to find points inside the frustum
        mask = (
            mask_w.flatten() &
            (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
            (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            # We skip strict Z clipping to avoid cutting off foreground/background too aggressively
            # unless desired. NDC Z is usually [-1, 1] or [0, 1].
        )
        
        subset_pts = original_pts[mask]
        subset_cols = original_cols[mask]
        
        # Subsample if necessary
        if len(subset_pts) > args.max_points:
            indices = np.random.choice(len(subset_pts), args.max_points, replace=False)
            subset_pts = subset_pts[indices]
            subset_cols = subset_cols[indices]
            
        print(f"Refining view: Found {mask.sum()} points in frustum, displaying {len(subset_pts)}")
        
        # Update plot
        # Remove old scatter plots
        for collection in ax.collections[:]:
            collection.remove()
            
        ax.scatter(subset_pts[:, 0], subset_pts[:, 1], subset_pts[:, 2], c=subset_cols, s=0.5, alpha=0.6, depthshade=True)
        
        # Update title
        if args.clean:
             title.set_text(f"Points: {len(subset_pts)} (Refined)")
        else:
             title.set_text(f"PLY Viewer ({len(subset_pts)} refined points)")
             
        fig.canvas.draw_idle()

    # Add button
    from matplotlib.widgets import Button
    ax_refine = plt.axes([0.8, 0.05, 0.1, 0.05])
    btn_refine = Button(ax_refine, 'Refine View')
    btn_refine.on_clicked(refine_view)
    
    # Debounce helper
    class Debounce:
        def __init__(self, timeout_ms, callback):
            self.timeout_ms = timeout_ms
            self.callback = callback
            self.timer = fig.canvas.new_timer(interval=timeout_ms)
            self.timer.add_callback(self._on_timer)

        def _on_timer(self):
            self.timer.stop()
            self.callback()

        def trigger(self):
            self.timer.stop()
            self.timer.start()

    debouncer = Debounce(500, refine_view)

    def on_scroll(event):
        # Matplotlib 3D removed ax.dist in recent versions.
        # We implement zoom by scaling the axis limits manually.
        base_scale = 1.1
        
        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Calculate centers and ranges
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        
        x_center = (xlim[0] + xlim[1]) / 2.0
        y_center = (ylim[0] + ylim[1]) / 2.0
        z_center = (zlim[0] + zlim[1]) / 2.0
        
        scale_factor = 1.0
        if event.button == 'up':
            # Zoom in (reduce range)
            scale_factor = 1.0 / base_scale
        elif event.button == 'down':
            # Zoom out (increase range)
            scale_factor = base_scale
            
        # Apply scaling
        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor
        new_z_range = z_range * scale_factor
        
        ax.set_xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)
        ax.set_ylim(y_center - new_y_range / 2, y_center + new_y_range / 2)
        ax.set_zlim(z_center - new_z_range / 2, z_center + new_z_range / 2)
        
        # Redraw immediately for smooth zoom
        fig.canvas.draw_idle()
        
        # Trigger refinement
        debouncer.trigger()

    def on_release(event):
        # Trigger on end of rotation/pan (left/right click release)
        debouncer.trigger()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_release_event', on_release)
    # Also bind key press 'r' for manual override
    def on_key(event):
        if event.key == 'r':
            refine_view()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial refine to make sure aspect ratio is applied correctly on first load if small
    # or just show initial subsample
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

