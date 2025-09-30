from pathlib import Path
from PIL import Image

def preprocess_images(input_dir: Path, output_dir: Path, target_size=(1280, 720), quality=80):
    """
    批量裁剪图片为 16:9 并缩放到指定大小，保存为压缩 JPG。
    
    Args:
        input_dir (Path): 输入图片文件夹
        output_dir (Path): 输出保存文件夹
        target_size (tuple): 目标分辨率 (W, H)，默认 (1280, 720)
        quality (int): JPEG 压缩质量 (1-100)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*.png"):
        img = Image.open(img_path)
        w, h = img.size
        target_ratio = target_size[0] / target_size[1]
        current_ratio = w / h

        # 根据宽高比裁剪成 16:9
        if current_ratio > target_ratio:  # 太宽 -> 裁左右
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            right = left + new_w
            top, bottom = 0, h
        else:  # 太高 -> 裁上下
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            bottom = top + new_h
            left, right = 0, w

        img_cropped = img.crop((left, top, right, bottom))

        # 缩放到目标分辨率
        img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)

        # 保存到输出目录
        out_path = output_dir / (img_path.stem + ".jpg")
        img_resized.save(out_path, format="JPEG", quality=quality, optimize=True)

        print(f"✅ Processed: {img_path.name} -> {out_path.name}")

# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    input_dir = Path("Frames/Raw")             # 原始图片目录
    output_dir = Path("Frames")  # 输出目录
    preprocess_images(input_dir, output_dir, target_size=(1280, 720), quality=80)
