import os
import requests
from json import JSONDecoder
import time
import argparse
import re 

# lib.utils.common_utils 假设存在，或者你不需要它
# from lib.utils.common_utils import get_labels_from_path

# ------------------ 恢复状态函数 ------------------
def load_processed_state(out_file):
    """从现有的输出文件中加载已处理的状态"""
    processed_files = set()
    total_confidences = []
    same_person_count = 0
    total_valid_count = 0

    if not os.path.exists(out_file):
        return processed_files, total_confidences, same_person_count, total_valid_count

    # 这个正则表达式 (.*?) 会自动捕获 'filename' 或 'subdir/filename'
    line_regex = re.compile(r"^(.*?)\s*\|\s*相似度:\s*([\d\.]+)")

    try:
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                match = line_regex.match(line)
                if match:
                    # 'filename' 变量现在会存储 '0001' 或 'age_10/0001'
                    filename = match.group(1).strip()
                    confidence = float(match.group(2))
                    
                    processed_files.add(filename)
                    total_confidences.append(confidence)
                    total_valid_count += 1
                    if confidence >= 70:
                        same_person_count += 1
                        
    except Exception as e:
        print(f"⚠️ 读取日志文件 {out_file} 出错: {e}。将从头开始。")
        return set(), [], 0, 0

    return processed_files, total_confidences, same_person_count, total_valid_count


# ------------------ 人脸比对函数 ------------------
def compareIm(face_path1, face_path2, api_key, api_secret):
    compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
    try:
        with open(face_path1, "rb") as f1, open(face_path2, "rb") as f2:
            files = {"image_file1": f1, "image_file2": f2}
            data = {"api_key": api_key, "api_secret": api_secret}
            response = requests.post(compare_url, data=data, files=files)
            response.raise_for_status()
            result_dict = JSONDecoder().decode(response.content.decode('utf-8'))
            return result_dict.get('confidence', None)
    except Exception as e:
        print(f"❌ 比对失败: {face_path1} vs {face_path2} | 错误: {e}")
        return None

# ------------------ 主流程函数 ------------------
def find_and_compare_images(args):
    
    file_mode = "a" # 默认是追加模式

    if args.restart:
        print("🔄 --restart 标志已设置。将从头开始评估并覆盖旧日志。")
        processed_files = set()
        total_confidences = []
        same_person_count = 0
        total_valid_count = 0
        file_mode = "w" # 覆盖模式
    else:
        # --- Checkpoint: 加载之前的状态 ---
        processed_files, total_confidences, same_person_count, total_valid_count = \
            load_processed_state(args.out_file)
        
        if processed_files:
            print(f"✅ 恢复状态: 已加载 {len(processed_files)} 个已处理过的文件。")
            if total_confidences:
                current_avg = sum(total_confidences) / len(total_confidences)
                print(f"   (当前累计平均值: {current_avg:.2f}, 累计有效比对: {total_valid_count})")
        else:
            print("ℹ️ 未找到先前的日志文件，将从头开始创建。")
        
        file_mode = "a" # 追加模式
    # --- 结束 Checkpoint ---

    # 使用 'a' (追加) 或 'w' (覆盖) 模式打开文件
    with open(args.out_file, file_mode, encoding="utf-8") as f_out:
        files_dir1 = {os.path.splitext(file)[0]: file for file in os.listdir(args.folder1)}

        # total_confidences 等变量已从上面加载或初始化

        for subdir_name in sorted(os.listdir(args.folder2)):
            print(f"\n--- 正在处理子目录: {subdir_name} ---") # 增加一个清晰的提示
            subdir_path = os.path.join(args.folder2, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            files_dir2 = {os.path.splitext(file)[0]: file for file in os.listdir(subdir_path)}
            common_files = set(files_dir1.keys()).intersection(set(files_dir2.keys()))
            
            confidences_this_run = []

            for idx, filename in enumerate(sorted(common_files)):
                if args.max_num and idx >= args.max_num:
                    print(f"已达到 {args.max_num} 个文件上限，跳到下一个子目录。")
                    break

                # --- MODIFIED 1: 创建唯一的ID ---
                # 使用 'subdir_name/filename' 作为唯一的键
                unique_id = f"{subdir_name}/{filename}"

                # --- MODIFIED 2: 使用 unique_id 检查 ---
                if unique_id in processed_files:
                    # print(f"跳过已处理: {unique_id}") # (可选) 打印跳过信息
                    continue

                path1 = os.path.join(args.folder1, files_dir1[filename])
                path2 = os.path.join(subdir_path, files_dir2[filename])

                confidence = compareIm(path1, path2, args.api_key, args.api_secret)

                if confidence is not None:
                    # --- MODIFIED 3: 使用 unique_id 记录 ---
                    log_str = f"{unique_id} | 相似度: {confidence:.2f}"
                    print(log_str)
                    f_out.write(log_str + "\n")
                    f_out.flush() # 立即写入磁盘，防止中断丢失数据
                    
                    confidences_this_run.append(confidence) 
                    total_confidences.append(confidence) 
                    total_valid_count += 1

                    if confidence >= 70:
                        same_person_count += 1
                    
                    processed_files.add(unique_id) # 将 unique_id 加入set
                else:
                    print(f"跳过无结果比对: {filename}")

                time.sleep(1.1)

            if confidences_this_run:
                avg = sum(confidences_this_run) / len(confidences_this_run)
                avg_str = f"[{subdir_name}] (本次运行) 平均相似度: {avg:.2f} (新增 {len(confidences_this_run)} 个)"
                print(avg_str)
                f_out.write(avg_str + "\n")
                f_out.flush()
            else:
                # 检查此子目录中是否有 *任何* 文件被处理过 (无论是本次还是之前)
                processed_in_this_subdir = any(f.startswith(subdir_name + '/') for f in processed_files)
                
                if not processed_in_this_subdir:
                     print(f"[{subdir_name}] 无有效比对")
                else:
                     print(f"[{subdir_name}] (本次运行) 无新增比对")


        if total_confidences:
            global_avg = sum(total_confidences) / len(total_confidences)
            final_str = f"\n[总计] 所有目录累计平均相似度: {global_avg:.2f} (共 {len(total_confidences)} 个)"
            print(final_str)
            f_out.write(final_str + "\n")

            same_person_ratio = same_person_count / total_valid_count if total_valid_count > 0 else 0
            same_person_str = f"[统计] 累计判定为同一个人的比例 (≥70分): {same_person_ratio:.2%}"
            print(same_person_str)
            f_out.write(same_person_str + "\n")
        else:
            print("⚠️ 没有任何有效的相似度结果")

# ------------------ 命令行参数定义 ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Face++ Identity Similarity Evaluation (with Checkpoint)")

    parser.add_argument('--folder1', type=str, default=None,
                        help='Directory containing original images')

    parser.add_argument('--folder2', type=str,
                        default=None,
                        help='Root directory containing aging results in subdirectories (per target age)')

    parser.add_argument('--out_file', type=str, default=None,
                        help='Path to output result file (default: result_<sep>.txt in dir2_root)')

    parser.add_argument('--max_num', type=int, default=200,
                        help='Maximum number of images to evaluate per directory')

    parser.add_argument('--api_key', type=str, default=None, help='Face++ API Key')
    parser.add_argument('--api_secret', type=str, default=None, help='Face++ API Secret')
    
    # --- 新增参数 ---
    parser.add_argument('--restart', action='store_true',
                        help='Restart evaluation from scratch, ignoring previous logs')
    # --- 结束 ---

    args = parser.parse_args()

    # 简单的参数校验
    if not all([args.folder1, args.folder2, args.api_key, args.api_secret]):
        parser.error("folder1, folder2, api_key, 和 api_secret 都是必需的参数。")
        
    if args.out_file is None:
        args.out_file = os.path.join(args.folder2, f"face_compare.txt")

    return args

# ------------------ 主程序入口 ------------------
if __name__ == '__main__':
    args = parse_args()
    find_and_compare_images(args)