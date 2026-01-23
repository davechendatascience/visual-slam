import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--html", default=None, help="Optional path to save HTML viewer")
    parser.add_argument("--normalize", action="store_true", help="Center and scale points for viewing")
    args = parser.parse_args()

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    # Use colab-friendly renderer when available
    try:
        import google.colab  # type: ignore
        pio.renderers.default = "colab"
    except Exception:
        pass

    points = []
    colors = []
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
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    if not points:
        raise RuntimeError("PLY file is empty or could not be read.")

    pts = np.array(points)
    cols = np.array(colors)

    if args.normalize and len(pts) > 0:
        center = pts.mean(axis=0)
        pts = pts - center
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=cols, opacity=0.8),
            )
        ]
    )
    fig.update_layout(scene=dict(aspectmode="data"), title="PLY Viewer")
    if args.html:
        fig.write_html(args.html)
        print(f"Saved viewer to {args.html}")
    fig.show()


if __name__ == "__main__":
    main()

