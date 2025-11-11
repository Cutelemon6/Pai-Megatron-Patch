# file: split_dataset.py
import sys
import os
import random

def split_jsonl_file(file_path, output_dir, seed=42):
    """
    读取一个 .jsonl 文件，将其随机划分为 90% 的训练集和 10% 的验证集，
    并写入到输出目录的新文件中。

    Args:
        file_path (str): 输入的 .jsonl 文件路径。
        output_dir (str): 输出目录。
        seed (int): 随机种子，用于保证划分结果可复现。
    """
    try:
        print(f"--- Processing file: {os.path.basename(file_path)} ---")

        # 1. 读取所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        if not lines:
            print("  File is empty. Skipping.")
            return

        # 2. 使用固定种子打乱行顺序，以保证每次运行结果一致
        random.seed(seed)
        random.shuffle(lines)

        # 3. 计算分割点
        total_lines = len(lines)
        # 确保即使在文件很小的情况下，验证集至少有1行（除非文件总共只有1行）
        split_index = max(1, int(total_lines * 0.10)) if total_lines > 1 else 0


        # 4. 分割数据
        valid_lines = lines[:split_index]
        train_lines = lines[split_index:]

        # 5. 构造输出文件名
        original_filename = os.path.basename(file_path)
        train_filename = os.path.join(output_dir, f"train-{original_filename}")
        valid_filename = os.path.join(output_dir, f"valid-{original_filename}")

        # 6. 写入训练集文件
        with open(train_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_lines))
            # 在文件末尾添加一个换行符，符合.jsonl常见格式
            if train_lines:
                f.write('\n')
        print(f"  ✓ Created training set: {os.path.basename(train_filename)} ({len(train_lines)} lines)")

        # 7. 写入验证集文件
        with open(valid_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines))
            if valid_lines:
                f.write('\n')
        print(f"  ✓ Created validation set: {os.path.basename(valid_filename)} ({len(valid_lines)} lines)")

    except Exception as e:
        print(f"  ✗ An error occurred: {e}")

def main():
    """
    主函数，处理命令行参数并遍历目录。
    """
    if len(sys.argv) != 2:
        print("Usage: python split_dataset.py <directory_path>")
        print("Example: python split_dataset.py /path/to/your/jsonl_files")
        sys.exit(1)

    input_dir = sys.argv[1]

    if not os.path.isdir(input_dir):
        print(f"Error: Provided path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    # 在输入目录旁边创建一个新的输出目录，以避免混乱
    output_dir = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}-split")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # 遍历目录下的所有 .jsonl 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_dir, filename)
            split_jsonl_file(file_path, output_dir)

if __name__ == "__main__":
    main()
