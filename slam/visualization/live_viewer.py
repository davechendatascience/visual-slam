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

