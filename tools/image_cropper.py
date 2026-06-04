#!/usr/bin/env python3
"""A small manual crop tool for importing large illustration sheets.

Usage:
  python tools/image_cropper.py [path/to/big_image.png]

The app lets you:
  - open a large source image
  - drag a crop rectangle on the canvas
  - choose a preset target
  - save the crop to one or more destinations at once

It is intentionally lightweight: Tkinter + Pillow only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Preset:
    label: str
    targets: Tuple[Path, ...]


def _p(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def _targets(*rel_paths: str) -> Tuple[Path, ...]:
    return tuple(_p(path) for path in rel_paths)


PRESETS: Sequence[Preset] = (
    Preset("Block 1 / 概览图", _targets("assets/images/block1_overview.png")),
    Preset("Block 1 / 补充图", _targets("assets/images/block1_supplement.png")),
    Preset(
        "Block 1 / 任务一 / 线性回归",
        _targets(
            "assets/images/mission_0.png",
            "exercises/block_01_basics/task_00_linear_regression/assets/mission_0.png",
        ),
    ),
    Preset(
        "Block 1 / 任务二 / 圆形分类",
        _targets(
            "assets/images/data_circle.png",
            "exercises/block_01_basics/task_01_circle_classifier/assets/data_circle.png",
        ),
    ),
    Preset(
        "Block 1 / 任务二 / ReLU",
        _targets(
            "assets/images/relu.png",
            "exercises/block_01_basics/task_01_circle_classifier/assets/relu.png",
        ),
    ),
    Preset(
        "Block 1 / 任务二 / Sigmoid",
        _targets(
            "assets/images/sigmoid.png",
            "exercises/block_01_basics/task_01_circle_classifier/assets/sigmoid.png",
        ),
    ),
    Preset(
        "Block 1 / 任务二 / 网络结构",
        _targets(
            "assets/images/relu_network_structure.png",
            "exercises/block_01_basics/task_01_circle_classifier/assets/relu_network_structure.png",
        ),
    ),
    Preset(
        "Block 1 / 任务二 / 近似曲线",
        _targets(
            "assets/images/relu_approximation.png",
            "exercises/block_01_basics/task_01_circle_classifier/assets/relu_approximation.png",
        ),
    ),
    Preset(
        "Block 1 / 任务三 / 库拆分",
        _targets(
            "assets/images/mini_dl_lib_split.png",
            "exercises/block_01_basics/task_02_mini_dl_lib/assets/mini_dl_lib_split.png",
        ),
    ),
    Preset(
        "Block 1 / 任务四 / MNIST MLP",
        _targets(
            "assets/images/mnist_mlp.png",
            "exercises/block_01_basics/task_03_mnist_mlp/assets/mnist_mlp.png",
        ),
    ),
    Preset("Block 1 / 训练闭环", _targets("assets/images/block1_training_loop.png")),
    Preset("Block 2 / 概览图", _targets("assets/images/block2_overview.png")),
    Preset(
        "Block 2 / 任务10 / 数据管线",
        _targets(
            "assets/images/image_data_pipeline.png",
            "exercises/block_02_resnet/task_10_image_data_pipeline/assets/image_data_pipeline.png",
        ),
    ),
    Preset(
        "Block 2 / 任务10 / NHWC-NCHW",
        _targets(
            "assets/images/nhwc_nchw.png",
            "exercises/block_02_resnet/task_10_image_data_pipeline/assets/nhwc_nchw.png",
        ),
    ),
    Preset(
        "Block 2 / 任务10 / 数据增强",
        _targets(
            "assets/images/data_augmentation.png",
            "exercises/block_02_resnet/task_10_image_data_pipeline/assets/data_augmentation.png",
        ),
    ),
    Preset(
        "Block 2 / 任务10 / 任务路线",
        _targets(
            "assets/images/resnet_task_route.png",
            "exercises/block_02_resnet/task_10_image_data_pipeline/assets/resnet_task_route.png",
        ),
    ),
    Preset(
        "Block 2 / 任务11 / 卷积解释",
        _targets(
            "assets/images/conv2d_explained.png",
            "exercises/block_02_resnet/task_11_conv2d_im2col/assets/conv2d_explained.png",
        ),
    ),
    Preset(
        "Block 2 / 任务11 / im2col",
        _targets(
            "assets/images/im2col_explained.png",
            "exercises/block_02_resnet/task_11_conv2d_im2col/assets/im2col_explained.png",
        ),
    ),
    Preset(
        "Block 2 / 任务11 / padding-stride",
        _targets(
            "assets/images/padding_stride.png",
            "exercises/block_02_resnet/task_11_conv2d_im2col/assets/padding_stride.png",
        ),
    ),
    Preset(
        "Block 2 / 任务12 / MaxPool",
        _targets(
            "assets/images/maxpool.png",
            "exercises/block_02_resnet/task_12_pooling_and_bn/assets/maxpool.png",
        ),
    ),
    Preset(
        "Block 2 / 任务12 / GlobalAvgPool",
        _targets(
            "assets/images/globalavgpool.png",
            "exercises/block_02_resnet/task_12_pooling_and_bn/assets/globalavgpool.png",
        ),
    ),
    Preset(
        "Block 2 / 任务12 / BatchNorm",
        _targets(
            "assets/images/batchnorm.png",
            "exercises/block_02_resnet/task_12_pooling_and_bn/assets/batchnorm.png",
        ),
    ),
    Preset(
        "Block 2 / 任务13 / 残差块",
        _targets(
            "assets/images/residual_block.png",
            "exercises/block_02_resnet/task_13_residual_block/assets/residual_block.png",
        ),
    ),
    Preset(
        "Block 2 / 任务13 / 梯度通路",
        _targets(
            "assets/images/residual_gradient_path.png",
            "exercises/block_02_resnet/task_13_residual_block/assets/residual_gradient_path.png",
        ),
    ),
    Preset(
        "Block 2 / 任务14 / ResNet 总图",
        _targets(
            "assets/images/resnet.png",
            "exercises/block_02_resnet/task_14_numpy_resnet_train/assets/resnet.png",
        ),
    ),
    Preset(
        "Block 2 / 任务15 / 分错样例",
        _targets(
            "assets/images/misclassified_examples.png",
            "exercises/block_02_resnet/task_15_experiment_notes/assets/misclassified_examples.png",
        ),
    ),
    Preset("Block 3 / 注意力基础概览", _targets("assets/images/block3_attention_overview.png")),
    Preset("Block 3 / 模型结构概览", _targets("assets/images/block3_model_overview.png")),
    Preset("Block 3 / 训练与生成概览", _targets("assets/images/block3_training_overview.png")),
    Preset(
        "Block 3 / 任务20 / Self-Attention",
        _targets(
            "assets/images/self_attention.png",
            "exercises/block_03_transformer/task_20_transformer_theory/assets/self_attention.png",
        ),
    ),
    Preset(
        "Block 3 / 任务20 / Encoder-Decoder",
        _targets(
            "assets/images/encoder_decoder_vs_decoder_only.png",
            "exercises/block_03_transformer/task_20_transformer_theory/assets/encoder_decoder_vs_decoder_only.png",
        ),
    ),
    Preset(
        "Block 3 / 任务21 / Sinusoidal",
        _targets(
            "assets/images/sinusoidal_position.png",
            "exercises/block_03_transformer/task_21_sinusoidal_position/assets/sinusoidal_position.png",
        ),
    ),
    Preset(
        "Block 3 / 任务21 / 位置接入",
        _targets(
            "assets/images/embedding_plus_position.png",
            "exercises/block_03_transformer/task_21_sinusoidal_position/assets/embedding_plus_position.png",
        ),
    ),
    Preset(
        "Block 3 / 任务22 / RoPE",
        _targets(
            "assets/images/rope.png",
            "exercises/block_03_transformer/task_22_rope_position/assets/rope.png",
        ),
    ),
    Preset(
        "Block 3 / 任务23 / Causal Mask",
        _targets(
            "assets/images/causal_mask.png",
            "exercises/block_03_transformer/task_23_causal_attention/assets/causal_mask.png",
        ),
    ),
    Preset(
        "Block 3 / 任务23 / MHA",
        _targets(
            "assets/images/mha.png",
            "exercises/block_03_transformer/task_23_causal_attention/assets/mha.png",
        ),
    ),
    Preset(
        "Block 3 / 任务23 / GQA",
        _targets(
            "assets/images/gqa.png",
            "exercises/block_03_transformer/task_23_causal_attention/assets/gqa.png",
        ),
    ),
    Preset(
        "Block 3 / 任务24 / SwiGLU",
        _targets(
            "assets/images/swiglu.png",
            "exercises/block_03_transformer/task_24_swiglu_ffn/assets/swiglu.png",
        ),
    ),
    Preset(
        "Block 3 / 任务25 / Embedding-LM Head",
        _targets(
            "assets/images/embedding_lm_head.png",
            "exercises/block_03_transformer/task_25_embedding_lm_head/assets/embedding_lm_head.png",
        ),
    ),
    Preset(
        "Block 3 / 任务26 / Decoder Block",
        _targets(
            "assets/images/decoder_block.png",
            "exercises/block_03_transformer/task_26_decoder_blocks/assets/decoder_block.png",
        ),
    ),
    Preset(
        "Block 3 / 任务27 / MiniMind 总图",
        _targets(
            "assets/images/minimind_overview.png",
            "exercises/block_03_transformer/task_27_minimind_core/assets/minimind_overview.png",
        ),
    ),
    Preset(
        "Block 3 / 任务28 / Shifted Labels",
        _targets(
            "assets/images/shifted_labels.png",
            "exercises/block_03_transformer/task_28_next_token_training/assets/shifted_labels.png",
        ),
    ),
    Preset(
        "Block 3 / 任务29 / Sampling",
        _targets(
            "assets/images/sampling_methods.png",
            "exercises/block_03_transformer/task_29_generate_sampling/assets/sampling_methods.png",
        ),
    ),
    Preset(
        "Block 3 / 任务30 / KV Cache",
        _targets(
            "assets/images/kv_cache.png",
            "exercises/block_03_transformer/task_30_kv_cache/assets/kv_cache.png",
        ),
    ),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ImageCropperApp:
    canvas_width = 980
    canvas_height = 720

    def __init__(self, root: tk.Tk, source: Optional[Path] = None) -> None:
        self.root = root
        self.root.title("教材裁图器")

        self.source_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.image_path: Optional[Path] = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selection_canvas: Optional[Tuple[int, int, int, int]] = None
        self.rect_id: Optional[int] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.current_preset = 0

        self.auto_advance = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_shortcuts()
        self._load_presets()

        if source is not None:
            self.load_source_image(source)

    def _build_ui(self) -> None:
        self.root.geometry("1400x900")

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="打开大图", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="保存到预设", command=self.save_to_preset).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="另存为...", command=self.save_as_custom).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="清除选区", command=self.clear_selection).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(toolbar, text="保存后自动切到下一个预设", variable=self.auto_advance).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, width=self.canvas_width, height=self.canvas_height, bg="#e9e9e9", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        right = ttk.Frame(body, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="预设目标", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        list_frame = ttk.Frame(right)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 8))

        self.preset_list = tk.Listbox(list_frame, height=18, exportselection=False)
        preset_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.preset_list.yview)
        self.preset_list.configure(yscrollcommand=preset_scroll.set)
        self.preset_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preset_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preset_list.bind("<<ListboxSelect>>", self.on_preset_select)

        info = ttk.LabelFrame(right, text="当前目标", padding=(10, 8))
        info.pack(fill=tk.X, pady=(0, 8))
        self.target_label = ttk.Label(info, text="先从左上角打开一张大图。", justify=tk.LEFT, wraplength=300)
        self.target_label.pack(anchor="w")

        self.preview_label = ttk.LabelFrame(right, text="选区预览", padding=(10, 8))
        self.preview_label.pack(fill=tk.X, pady=(0, 8))
        self.preview_canvas = tk.Canvas(self.preview_label, width=300, height=180, bg="#f4f4f4", highlightthickness=0)
        self.preview_canvas.pack()

        selection_frame = ttk.LabelFrame(right, text="选区信息", padding=(10, 8))
        selection_frame.pack(fill=tk.X)
        self.status_text = tk.StringVar(value="未加载图片")
        ttk.Label(selection_frame, textvariable=self.status_text, justify=tk.LEFT, wraplength=300).pack(anchor="w")

        self.source_text = tk.StringVar(value="源图：未加载")
        ttk.Label(right, textvariable=self.source_text, foreground="#555").pack(anchor="w", pady=(10, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-s>", lambda _e: self.save_to_preset())
        self.root.bind("<Escape>", lambda _e: self.clear_selection())
        self.root.bind("<Right>", lambda _e: self.select_preset(self.current_preset + 1))
        self.root.bind("<Left>", lambda _e: self.select_preset(self.current_preset - 1))

    def _load_presets(self) -> None:
        self.preset_list.delete(0, tk.END)
        for preset in PRESETS:
            self.preset_list.insert(tk.END, preset.label)
        if PRESETS:
            self.preset_list.selection_set(0)
            self.preset_list.activate(0)
            self.update_target_info(0)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择要裁切的大图",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_source_image(Path(path))

    def load_source_image(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGBA")
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("打开失败", f"无法打开图片：{path}\n\n{exc}")
            return

        self.source_image = image
        self.image_path = path
        self.source_text.set(f"源图：{path}")
        self.clear_selection()
        self._fit_image_to_canvas()
        self.draw_image()
        self.status_text.set(f"已加载 {path.name}，可以开始框选。")

    def _fit_image_to_canvas(self) -> None:
        if self.source_image is None:
            return
        img_w, img_h = self.source_image.size
        scale = min(self.canvas_width / img_w, self.canvas_height / img_h)
        scale = min(scale, 1.0)
        self.scale = scale
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))
        self.display_image = self.source_image.resize((disp_w, disp_h), Image.LANCZOS)
        self.offset_x = (self.canvas_width - disp_w) // 2
        self.offset_y = (self.canvas_height - disp_h) // 2

    def draw_image(self) -> None:
        self.canvas.delete("all")
        self.rect_id = None
        if self.display_image is None:
            return
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo_image)
        self.canvas.create_rectangle(
            self.offset_x,
            self.offset_y,
            self.offset_x + self.display_image.width,
            self.offset_y + self.display_image.height,
            outline="#999",
        )
        self.redraw_selection()

    def clear_selection(self) -> None:
        self.selection_canvas = None
        self.drag_start = None
        self.rect_id = None
        self.preview_canvas.delete("all")
        if self.source_image is not None:
            self.status_text.set("没有选区。按住鼠标左键拖出一个裁切区域。")

    def on_press(self, event: tk.Event) -> None:
        if self.source_image is None:
            return
        if not self._point_in_image(event.x, event.y):
            return
        self.drag_start = (event.x, event.y)
        self.selection_canvas = (event.x, event.y, event.x, event.y)
        self.redraw_selection()

    def on_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.selection_canvas = (x0, y0, event.x, event.y)
        self.redraw_selection()

    def on_release(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.selection_canvas = (x0, y0, event.x, event.y)
        self.drag_start = None
        self.redraw_selection()
        self.update_preview_and_status()

    def redraw_selection(self) -> None:
        self.canvas.delete("selection")
        if self.selection_canvas is None:
            return
        x1, y1, x2, y2 = self.selection_canvas
        x1, y1, x2, y2 = self._clamp_canvas_rect(x1, y1, x2, y2)
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff5a00", width=2, tags="selection")

    def _point_in_image(self, cx: int, cy: int) -> bool:
        if self.display_image is None:
            return False
        x1 = self.offset_x
        y1 = self.offset_y
        x2 = x1 + self.display_image.width
        y2 = y1 + self.display_image.height
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _clamp_canvas_rect(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        if self.display_image is None:
            return x1, y1, x2, y2
        left = self.offset_x
        top = self.offset_y
        right = left + self.display_image.width
        bottom = top + self.display_image.height
        x1 = int(clamp(x1, left, right))
        x2 = int(clamp(x2, left, right))
        y1 = int(clamp(y1, top, bottom))
        y2 = int(clamp(y2, top, bottom))
        return x1, y1, x2, y2

    def canvas_rect_to_image_box(self) -> Optional[Tuple[int, int, int, int]]:
        if self.source_image is None or self.selection_canvas is None:
            return None
        x1, y1, x2, y2 = self._clamp_canvas_rect(*self.selection_canvas)
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if right - left < 2 or bottom - top < 2:
            return None

        img_left = int(round((left - self.offset_x) / self.scale))
        img_right = int(round((right - self.offset_x) / self.scale))
        img_top = int(round((top - self.offset_y) / self.scale))
        img_bottom = int(round((bottom - self.offset_y) / self.scale))

        img_w, img_h = self.source_image.size
        img_left = int(clamp(img_left, 0, img_w))
        img_right = int(clamp(img_right, 0, img_w))
        img_top = int(clamp(img_top, 0, img_h))
        img_bottom = int(clamp(img_bottom, 0, img_h))

        if img_right - img_left < 1 or img_bottom - img_top < 1:
            return None
        return img_left, img_top, img_right, img_bottom

    def update_preview_and_status(self) -> None:
        box = self.canvas_rect_to_image_box()
        if box is None:
            self.preview_canvas.delete("all")
            self.status_text.set("选区太小，至少拖出一个有效区域。")
            return
        assert self.source_image is not None
        crop = self.source_image.crop(box)
        preview = crop.copy()
        preview.thumbnail((290, 170), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(150, 90, image=self._preview_photo, anchor=tk.CENTER)
        self.status_text.set(
            f"当前选区：canvas={self.selection_canvas} | image={box} | size={crop.size[0]}x{crop.size[1]}"
        )

    def on_preset_select(self, _event: tk.Event) -> None:
        selection = self.preset_list.curselection()
        if not selection:
            return
        self.update_target_info(selection[0])

    def update_target_info(self, index: int) -> None:
        index = int(clamp(index, 0, len(PRESETS) - 1))
        self.current_preset = index
        self.preset_list.selection_clear(0, tk.END)
        self.preset_list.selection_set(index)
        self.preset_list.see(index)

        preset = PRESETS[index]
        lines = [preset.label, ""]
        for target in preset.targets:
            lines.append(str(target.relative_to(REPO_ROOT)))
        self.target_label.configure(text="\n".join(lines))

    def select_preset(self, index: int) -> None:
        if not PRESETS:
            return
        index = index % len(PRESETS)
        self.update_target_info(index)

    def _save_crop_to_targets(self, targets: Iterable[Path]) -> None:
        assert self.source_image is not None
        box = self.canvas_rect_to_image_box()
        if box is None:
            messagebox.showwarning("没有选区", "请先拖出一个有效的裁切区域。")
            return

        crop = self.source_image.crop(box).convert("RGBA")
        target_list = [Path(target) for target in targets]
        if not target_list:
            messagebox.showwarning("没有目标", "当前预设没有配置输出路径。")
            return

        existing = [target for target in target_list if target.exists()]
        if existing:
            names = "\n".join(str(path.relative_to(REPO_ROOT)) for path in existing)
            if not messagebox.askyesno("确认覆盖", f"下面这些文件已存在，要覆盖吗？\n\n{names}"):
                return

        for target in target_list:
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)

        saved = "\n".join(str(target.relative_to(REPO_ROOT)) for target in target_list)
        self.status_text.set(f"已保存到：\n{saved}")
        messagebox.showinfo("保存成功", f"已保存到：\n\n{saved}")

        if self.auto_advance.get() and PRESETS:
            self.select_preset(self.current_preset + 1)

    def save_to_preset(self) -> None:
        if self.source_image is None:
            messagebox.showwarning("还没打开图片", "请先打开一张大图。")
            return
        self._save_crop_to_targets(PRESETS[self.current_preset].targets)

    def save_as_custom(self) -> None:
        if self.source_image is None:
            messagebox.showwarning("还没打开图片", "请先打开一张大图。")
            return
        path = filedialog.asksaveasfilename(
            title="另存为",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("WEBP", "*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._save_crop_to_targets([Path(path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual cropper for lesson illustrations.")
    parser.add_argument("source", nargs="?", help="Optional path to the source image to open at startup.")
    args = parser.parse_args()

    root = tk.Tk()
    app = ImageCropperApp(root, Path(args.source) if args.source else None)
    root.mainloop()


if __name__ == "__main__":
    main()
