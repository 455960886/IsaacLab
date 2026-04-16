#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse


def read_text_auto(path: Path) -> str:
    encodings = ["utf-16", "utf-8", "utf-8-sig", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeError as e:
            last_err = e
    raise RuntimeError(f"无法解码文件: {path}") from last_err


def parse_points(text: str):
    points = []
    skipped = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            skipped += 1
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            q = float(parts[3]) if len(parts) >= 4 else 0.0
            points.append((x, y, z, q))
        except ValueError:
            skipped += 1

    return points, skipped


def write_ply(points, out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float quality\n")
        f.write("end_header\n")

        for x, y, z, q in points:
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {q:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert point cloud txt to MeshLab-compatible ASCII PLY.")
    parser.add_argument("input_txt", help="输入 txt 文件路径")
    parser.add_argument("-o", "--output", help="输出 ply 文件路径；不填则自动同名输出")
    args = parser.parse_args()

    in_path = Path(args.input_txt)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".ply")

    text = read_text_auto(in_path)
    points, skipped = parse_points(text)

    if not points:
        raise RuntimeError("没有解析到任何有效点。请检查 txt 文件格式是否为: x y z [optional_value]")

    write_ply(points, out_path)

    print(f"转换完成: {out_path}")
    print(f"有效点数: {len(points)}")
    print(f"跳过行数: {skipped}")


if __name__ == "__main__":
    main()