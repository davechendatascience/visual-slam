import threading
import time

import numpy as np


class LiveMapViewer:
    def __init__(self, gaussian_map, map_lock, interval_s=1.0, max_points=200000):
        self.gaussian_map = gaussian_map
        self.map_lock = map_lock
        self.interval_s = interval_s
        self.max_points = max_points
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _loop(self):
        try:
            import google.colab  # type: ignore
            in_colab = True
        except Exception:
            in_colab = False

        if in_colab:
            try:
                from IPython.display import clear_output, display
                import plotly.graph_objects as go
                import plotly.io as pio
            except ImportError as exc:
                raise RuntimeError("plotly is required for live visualization in Colab.") from exc
            pio.renderers.default = "colab"
        else:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise RuntimeError("matplotlib is required for live visualization.") from exc

            plt.ion()
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter([], [], [], s=2)

        while not self.stop_event.is_set():
            with self.map_lock:
                means = self.gaussian_map.means.detach().cpu().numpy()
                colors = self.gaussian_map.colors.detach().cpu().numpy()
            if means.shape[0] > 0:
                if means.shape[0] > self.max_points:
                    idx = np.random.choice(means.shape[0], self.max_points, replace=False)
                    means = means[idx]
                    colors = colors[idx]
                if in_colab:
                    fig = go.Figure(
                        data=[
                            go.Scatter3d(
                                x=means[:, 0],
                                y=means[:, 1],
                                z=means[:, 2],
                                mode="markers",
                                marker=dict(size=2, color=colors, opacity=0.8),
                            )
                        ]
                    )
                    fig.update_layout(scene=dict(aspectmode="data"), title="Live Gaussian Map")
                    clear_output(wait=True)
                    display(fig)
                else:
                    scatter._offsets3d = (means[:, 0], means[:, 1], means[:, 2])
                    scatter.set_color(colors.clip(0.0, 1.0))
                    ax.set_title("Live Gaussian Map")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            time.sleep(self.interval_s)

