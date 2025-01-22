import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import torch
import matplotlib
matplotlib.use("Agg")  # 혹시 모를 백엔드 충돌 방지용(필수 아님)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt

import threading
import time
import timeit
import os
from datetime import datetime

# ---------------------------
# 기본 파라미터 설정
# ---------------------------
# (x0, y0): 플롯 중심 좌표
x0, y0 = 3, 3.3
# eps: x0를 중심으로 좌우로 eps만큼 표시 (16:9 비율)
eps = 0.0013
eps_y = eps * (9 / 16)
eps_threshold = 1e-4
n = 3080
nx, ny = n, int(n * (9 / 16))

#----------------------------
# 커스텀 파라미터
#---------------------------
base = 2  # z(n+1) = sqrt({base})^z(n) + c 에서 base
max_iter = 1000  # 최대 반복 횟수
escape_radius = 1e+10  # 발산 판정 값값


# ---------------------------
# 1) Torch로 "몇 번 만에 발산?" 계산
# ---------------------------
def compute_tetration_steps_torch(
    nx, ny, max_iter, escape_radius, x0, y0, eps, eps_y,
    progress_callback=None,
    threshold=1e-6
):
    """
    PyTorch로 a(n+1) = base^(a(n)) + c 계산 (복소수).
    - 이번에는 '발산 시점(몇 번째 반복에 escape?)'을 기록하는 배열을 반환한다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    # 1) 좌표 생성
    x = torch.linspace(x0 - eps, x0 + eps, nx, device=device)
    y = torch.linspace(y0 - eps_y, y0 + eps_y, ny, device=device)

    # 2) dtype: 이중 정밀도로 설정
    complex_dtype = torch.complex128

    # 3) c = x + i*y, shape=(nx, ny)
    X = x.view(-1, 1).float()  # (nx, 1)
    Y = y.view(1, -1).float()  # (1, ny)
    c = X + 1j * Y
    c = c.to(complex_dtype)

    # 4) base, log(base)
    base_real = torch.sqrt(torch.tensor(base, device=device, dtype=torch.float32))
    log_base = torch.log(base_real).to(complex_dtype)

    # 5) a(0) = base (복소형)
    a = torch.full_like(c, fill_value=base_real, dtype=complex_dtype)

    # 6) 발산 여부 & 발산 시점
    #    - not_done: 아직 발산 안 된 지점
    #    - steps: 발산 발생한 반복 횟수 (초기값 = max_iter)
    not_done = torch.ones(c.shape, dtype=torch.bool, device=device)
    divergence_steps = torch.full(c.shape, max_iter, dtype=torch.int32, device=device)

    update_frequency = max_iter // 10 if max_iter > 0 else 1

    for i in range(max_iter):
        a_old = a.clone()

        # a(n+1) = base^(a(n)) + c
        # Torch에서 base^z 미지원 → exp(a(n) * log(base))
        new_val = torch.exp(a_old * log_base) + c 

        # 아직 발산 안 된 지점만 업데이트
        a = torch.where(not_done, new_val, a)

        # 발산 체크
        diverge_mask = (torch.abs(a) > escape_radius) & not_done

        # 이번 i번째 반복에서 새로 발산한 지점
        divergence_steps[diverge_mask] = i  # i번째에 발산
        not_done[diverge_mask] = False

        # 수렴 체크 (선택 사항)
        diff = torch.abs(a - a_old)
        converge_mask = (diff < threshold) & not_done
        # (여기서는 '수렴'은 발산 시간 기록과 관계 없으므로, 굳이 업데이트 X)
        not_done[converge_mask] = False

        # 진행 상황 콜백
        if progress_callback and (i % update_frequency == 0 or i == max_iter - 1):
            progress_callback(i + 1, max_iter)

        # 남은 점이 없으면 조기 종료
        if not torch.any(not_done):
            break

    # CPU로 복사 → numpy
    divergence_steps_cpu = divergence_steps.detach().cpu().numpy()

    total_time = time.time() - start_time
    print(f"[PyTorch] Total execution time: {total_time:.3f} seconds")

    return divergence_steps_cpu


# ---------------------------
# 2) 플롯(이미지) 생성 함수
# ---------------------------
def plot_tetration(app, ax, x0, y0, eps, eps_y):
    start_time = time.time()

    def progress_callback(current, total):
        progress = (current / total) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / current) * total if current > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        iterations_per_second = current / elapsed_time if elapsed_time > 0 else 0
        time_info = (f"Time: {remaining_time:.2f}s / Est.: {estimated_total_time:.2f}s, "
                     f"Progress: {progress:.2f}%, Speed: {iterations_per_second:.2f} it./s")
        app.status_label.config(text=f"Calculating... {time_info}")
        app.progress["value"] = progress
        app.root.update_idletasks()

    app.set_calculation_status(True)

    # [중요] 발산 시점(몇 번째 반복) 배열 계산
    divergence_steps = compute_tetration_steps_torch(
        nx, ny, max_iter, escape_radius,
        x0, y0, eps, eps_y,
        progress_callback=progress_callback,
        threshold=1e-6,
        eps_threshold=eps_threshold
    )

    # divergence_steps: 각 지점이 몇 번째에 발산했는지 (발산 안 하면 max_iter 유지)
    # → 이를 색으로 표시하기 위해 0 ~ max_iter 범위를 가진다.

    # 색상 맵 예: 'plasma', 'viridis' 등
    cmap = 'plasma'
    norm = plt.Normalize(vmin=0, vmax=max_iter)

    # GUI에서 on_move 시 픽셀 값(몇 번째에 발산)을 확인하기 위해 저장
    app.divergence_steps = divergence_steps
    app.extent = [x0 - eps, x0 + eps, y0 - eps_y, y0 + eps_y]

    ax.clear()
    ax.imshow(divergence_steps.T, extent=app.extent, origin='lower', cmap=cmap, norm=norm)

    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    ax.set_title(f"Tetration Plot (escape steps)\n x={x0}, y={y0}, eps={eps}")
    ax.set_autoscale_on(False)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))

    app.set_calculation_status(False)
    app.progress["value"] = 0


# ---------------------------
# 3) GUI + 이벤트 처리
# ---------------------------
clicks = []
rect = None

def on_click(event, app):
    global clicks, rect
    if event.inaxes == app.ax:
        clicks.append((event.xdata, event.ydata))
        if len(clicks) == 2:
            app.status_label.config(text="Calculating...")
            app.update_plot()
        elif len(clicks) == 1:
            rect = app.ax.add_patch(
                plt.Rectangle(clicks[0], 0, 0, linewidth=2, edgecolor='r', facecolor='none')
            )
            app.canvas.draw()

def on_move(event, app):
    global rect
    # 빨간 사각형 그리기 (줌인 표시)
    if rect and len(clicks) == 1 and event.inaxes == app.ax:
        x0, y0 = clicks[0]
        x1, y1 = event.xdata, event.ydata
        rect.set_width(x1 - x0)
        rect.set_height(y1 - y0)
        rect.set_xy((x0, y0))
        app.canvas.draw()

    # 마우스 좌표에 해당하는 '발산 반복 횟수' 표시
    if event.inaxes == app.ax and app.divergence_steps is not None and app.extent is not None:
        xx, yy = event.xdata, event.ydata
        xmin, xmax, ymin, ymax = app.extent
        if xmin <= xx <= xmax and ymin <= yy <= ymax:
            # 픽셀 좌표로 환산
            px = int((xx - xmin) / (xmax - xmin) * app.divergence_steps.shape[0])
            py = int((yy - ymin) / (ymax - ymin) * app.divergence_steps.shape[1])
            if 0 <= px < app.divergence_steps.shape[0] and 0 <= py < app.divergence_steps.shape[1]:
                steps = app.divergence_steps[px, py]
                app.status_label.config(
                    text=f"Escape steps at ({xx:.6f}, {yy:.6f}) = {steps}"
                )


class ZoomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zoom Application: Torch + Escape Steps")
        self.calculation_in_progress = False
        self.invert_zoom = tk.BooleanVar(value=False)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.zoom_out_label = ttk.Label(self.toolbar_frame, text="Zoom Out: ")
        self.zoom_out_label.pack(side=tk.LEFT)

        self.invert_zoom_checkbox = ttk.Checkbutton(
            self.toolbar_frame, text="Invert Zoom",
            variable=self.invert_zoom, command=self.toggle_invert_zoom
        )
        self.invert_zoom_checkbox.pack(side=tk.LEFT)

        self.zoom_factors = [10, 100, 1000, 10000, 100000]
        self.buttons = []
        for factor in self.zoom_factors:
            button = ttk.Button(
                self.toolbar_frame,
                text=f"1/{factor}",
                command=lambda f=factor: self.zoom(f)
            )
            button.pack(side=tk.LEFT)
            self.buttons.append(button)

        self.status_label = tk.Label(self.root, text="Program Status: Ready")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=300, mode="determinate"
        )
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)

        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(self.controls_frame, text="X:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(self.controls_frame)
        self.x_entry.pack(side=tk.LEFT)

        tk.Label(self.controls_frame, text="Y:").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(self.controls_frame)
        self.y_entry.pack(side=tk.LEFT)

        tk.Label(self.controls_frame, text="Eps:").pack(side=tk.LEFT)
        self.eps_entry = ttk.Entry(self.controls_frame, width=25)
        self.eps_entry.pack(side=tk.LEFT)

        self.apply_button = ttk.Button(
            self.controls_frame, text="Apply", command=self.update_plot_from_entry
        )
        self.apply_button.pack(side=tk.LEFT)

        self.copy_button = ttk.Button(
            self.controls_frame, text="Copy", command=self.copy_to_clipboard
        )
        self.copy_button.pack(side=tk.LEFT)

        self.save_button = ttk.Button(
            self.controls_frame, text="Save\n.png", command=self.save_image
        )
        self.save_button.pack(side=tk.RIGHT)

        # 이벤트 연결
        self.cid_click = self.canvas.mpl_connect(
            'button_press_event', lambda event: on_click(event, self)
        )
        self.cid_move = self.canvas.mpl_connect(
            'motion_notify_event', lambda event: on_move(event, self)
        )

        self.current_x0 = x0
        self.current_y0 = y0
        self.current_eps = eps

        # 새롭게 추가된 변수
        self.divergence_steps = None
        self.extent = None

        self.plot_initial()

    def set_calculation_status(self, status, message=None):
        self.calculation_in_progress = status
        if status:
            self.status_label.config(text="Calculating...")
        else:
            if message is not None:
                self.status_label.config(text=message)
            else:
                self.status_label.config(text="Click two points to Zoom In.")

    def plot_initial(self):
        threading.Thread(
            target=self.update_plot_with_thread,
            args=(x0, y0, eps, eps_y)
        ).start()

    def toggle_invert_zoom(self):
        if self.invert_zoom.get():
            self.zoom_out_label.config(text="Zoom In: ")
            for i, factor in enumerate(self.zoom_factors):
                self.buttons[i].config(text=str(factor))
        else:
            self.zoom_out_label.config(text="Zoom Out: ")
            for i, factor in enumerate(self.zoom_factors):
                self.buttons[i].config(text=f"1/{factor}")

    def zoom(self, factor):
        self.set_calculation_status(True)
        self.progress["value"] = 0
        global x0, y0, eps, eps_y

        if self.invert_zoom.get():
            # Zoom In (사각형 표시)
            new_eps = eps / factor
            new_eps_y = eps_y / factor
            rect = plt.Rectangle(
                (x0 - new_eps, y0 - new_eps_y),
                2 * new_eps, 2 * new_eps_y,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            self.ax.add_patch(rect)
            self.canvas.draw()
            eps = new_eps
            eps_y = new_eps_y
        else:
            # Zoom Out
            eps *= factor
            eps_y *= factor

        threading.Thread(
            target=self.update_plot_with_thread,
            args=(x0, y0, eps, eps_y)
        ).start()

    def update_entries(self):
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(tk.END, str(x0))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(tk.END, str(y0))
        self.eps_entry.delete(0, tk.END)
        self.eps_entry.insert(tk.END, str(eps))

    def update_plot_from_entry(self):
        self.set_calculation_status(True, "Calculation in progress...(update_plot_from_entry)")
        self.progress["value"] = 0
        self.canvas.draw()
        global x0, y0, eps, eps_y
        try:
            new_x0 = float(self.x_entry.get())
            new_y0 = float(self.y_entry.get())
            new_eps = float(self.eps_entry.get())
            if (new_x0, new_y0, new_eps) == (self.current_x0, self.current_y0, self.current_eps):
                self.status_label.config(text="Parameters are the same. No need to update.")
                return
            x0, y0, eps = new_x0, new_y0, new_eps
            eps_y = eps * (9 / 16)
            threading.Thread(
                target=self.update_plot_with_thread,
                args=(x0, y0, eps, eps_y)
            ).start()
            self.current_x0 = x0
            self.current_y0 = y0
            self.current_eps = eps
        except ValueError as error:
            print("Error:", error)
            self.status_label.config(text="Invalid input. Please enter valid numbers.")

    def update_plot(self):
        self.set_calculation_status(True, "Calculation in progress...(update_plot)")
        self.progress["value"] = 0
        self.canvas.draw()
        global x0, y0, eps, eps_y, clicks, rect
        (x1, y1), (x2, y2) = clicks
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        eps = abs(x2 - x1) / 2
        eps_y = abs(y2 - y1) / 2
        threading.Thread(
            target=self.update_plot_with_thread,
            args=(x0, y0, eps, eps_y)
        ).start()
        clicks = []
        rect = None
        self.update_entries()
        self.current_x0 = x0
        self.current_y0 = y0
        self.current_eps = eps

    def update_plot_with_thread(self, x0, y0, eps, eps_y):
        plot_tetration(self, self.ax, x0, y0, eps, eps_y)
        self.set_calculation_status(False)
        self.status_label.config(text="Click two points to Zoom In.")
        self.canvas.draw()

    def copy_to_clipboard(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(f"x={x0}, y={y0}, eps={eps}")
        self.status_label.config(text="Copied to clipboard!")

    def save_image(self):
        today_date = datetime.today().strftime("%Y-%m-%d")
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, today_date)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = f"a0tetanokusc{self.current_x0}_y{self.current_y0}_eps{self.current_eps}_iter{max_iter}_base{base}.png"
        try:
            filepath = os.path.join(folder_path, filename)
            self.figure.savefig(filepath, bbox_inches='tight')
            self.status_label.config(text="Image saved successfully at: " + filepath)
        except Exception as e:
            self.status_label.config(text=f"Error occurred while saving: {str(e)}")


# ---------------------------
# 메인 실행부
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ZoomApp(root)
    root.mainloop()
