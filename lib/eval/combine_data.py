import os
import shutil
import argparse

def copy_all_subfolders(source_dir, dest_dir, limit=None):
    """
    将源目录中的前 N 个子文件夹复制到目标目录。

    参数:
    source_dir (str): 源文件夹路径。
    dest_dir (str): 目标文件夹路径。
    limit (int, 可选): 限制复制的子文件夹数量，默认为全部复制。
    """
    # 确保目标文件夹存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"创建了目标文件夹: {dest_dir}")

    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 错误: 源文件夹 {source_dir} 不存在。")
        return

    # 获取源文件夹下所有子文件夹
    try:
        subfolders = [f.name for f in os.scandir(source_dir) if f.is_dir()]
    except OSError as e:
        print(f"❌ 错误: 无法访问 {source_dir}: {e}")
        return

    if not subfolders:
        print(f"⚠ 源目录 {source_dir} 中没有子文件夹可复制。")
        return

    # 如果设置了限制数量
    if limit is not None:
        subfolders = subfolders[:limit]
        print(f"📋 仅复制前 {limit} 个子文件夹。")

    print(f"📁 共发现 {len(subfolders)} 个子文件夹，准备复制:")
    print(subfolders)

    # 循环复制子文件夹
    for folder_name in subfolders:
        source_path = os.path.join(source_dir, folder_name)
        destination_path = os.path.join(dest_dir, folder_name)

        if os.path.exists(destination_path):
            print(f"⏩ 跳过: '{folder_name}' 在目标位置已存在。")
            continue

        try:
            shutil.copytree(source_path, destination_path)
            print(f"✅ 成功复制 '{folder_name}' 到 '{dest_dir}'")
        except Exception as e:
            print(f"❌ 复制 '{folder_name}' 时出错: {e}")

    print("\n🎉 全部文件夹复制完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复制所有子文件夹工具")
    parser.add_argument("--source", required=True, help="源目录路径（包含子文件夹）")
    parser.add_argument("--dest", required=True, help="目标目录路径")
    parser.add_argument("--limit", type=int, default=3000, help="限制复制的子文件夹数量（默认复制全部）")

    args = parser.parse_args()

    copy_all_subfolders(args.source, args.dest, args.limit)
