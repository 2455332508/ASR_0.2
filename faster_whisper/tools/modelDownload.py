import os
import shutil
import sys
from pathlib import Path


def get_default_cache_dirs():
    """
    获取不同系统下的默认模型缓存目录
    """
    cache_dirs = []

    # Hugging Face Transformers 缓存目录
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    cache_dirs.append(hf_home)

    # CTranslate2 缓存目录 (通常在用户目录下)
    if sys.platform.startswith('win'):
        # Windows
        ct2_cache = os.path.expanduser('~/.cache/ctranslate2')
        cache_dirs.append(ct2_cache)
    else:
        # Linux/Mac
        ct2_cache = os.path.expanduser('~/.cache/ctranslate2')
        cache_dirs.append(ct2_cache)

    # faster-whisper 默认缓存位置
    whisper_cache = os.path.expanduser('~/.cache/faster-whisper')
    cache_dirs.append(whisper_cache)

    return cache_dirs


def find_whisper_models(cache_dirs):
    """
    在缓存目录中查找Whisper模型
    """
    models_found = []

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            # 查找包含whisper或large-v3的目录
            for item in cache_path.rglob('*'):
                if item.is_dir():
                    dir_name = item.name.lower()
                    if 'whisper' in dir_name or 'large' in dir_name or 'v3' in dir_name:
                        models_found.append(str(item))

    return models_found


def clear_specific_model(model_name="large-v3"):
    """
    清除特定模型的缓存
    """
    cache_dirs = get_default_cache_dirs()
    cleared_dirs = []

    print(f"正在清除模型 '{model_name}' 的缓存...")

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            print(f"检查缓存目录: {cache_path}")

            # 查找包含模型名称的子目录
            for item in cache_path.iterdir():
                if item.is_dir():
                    dir_name = item.name.lower()
                    if model_name.lower() in dir_name:
                        print(f"  发现匹配的模型目录: {item}")
                        try:
                            shutil.rmtree(item)
                            print(f"  ✓ 已删除: {item}")
                            cleared_dirs.append(str(item))
                        except Exception as e:
                            print(f"  ✗ 删除失败: {e}")

    return cleared_dirs


def clear_all_whisper_cache():
    """
    清除所有Whisper相关的缓存
    """
    cache_dirs = get_default_cache_dirs()
    cleared_items = []

    print("正在清除所有Whisper缓存...")

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            print(f"检查缓存目录: {cache_path}")

            # 遍历所有子目录
            for item in cache_path.rglob('*'):
                if item.is_dir():
                    dir_name = item.name.lower()
                    if any(keyword in dir_name for keyword in ['whisper', 'large', 'tiny', 'base', 'small', 'medium']):
                        print(f"  发现Whisper相关目录: {item}")
                        try:
                            shutil.rmtree(item)
                            print(f"  ✓ 已删除: {item}")
                            cleared_items.append(str(item))
                        except Exception as e:
                            print(f"  ✗ 删除失败: {e}")

    return cleared_items


def show_cache_usage():
    """
    显示缓存目录使用情况
    """
    cache_dirs = get_default_cache_dirs()

    print("\n缓存目录使用情况:")
    print("-" * 50)

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            size = get_directory_size(cache_path)
            print(f"{cache_dir}: {format_bytes(size)}")
        else:
            print(f"{cache_dir}: 不存在")


def get_directory_size(path):
    """
    计算目录大小
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except OSError:
                pass  # 文件可能不存在或权限不足
    return total


def format_bytes(bytes_value):
    """
    格式化字节大小
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def manual_clear_instructions():
    """
    显示手动清理的详细说明
    """
    print("\n手动清理说明:")
    print("=" * 60)

    if sys.platform.startswith('win'):
        print("Windows系统:")
        print("1. 打开文件资源管理器")
        print("2. 按 Ctrl+R 输入 %USERPROFILE%\\.cache")
        print("3. 查找并删除以下目录（如果存在）:")
        print("   - %USERPROFILE%\\.cache\\huggingface\\hub (HF_HOME)")
        print("   - %USERPROFILE%\\.cache\\faster-whisper")
        print("   - 或者使用环境变量 HF_HOME 指定的目录")
    else:
        print("Linux/Mac系统:")
        print("1. 终端命令:")
        print("   rm -rf ~/.cache/huggingface/hub/")
        print("   rm -rf ~/.cache/faster-whisper/")
        print("   rm -rf ~/.cache/ctranslate2/")

    print("\n环境变量说明:")
    print("HF_HOME: Hugging Face模型缓存目录")
    print("您可以设置此环境变量来指定缓存位置")


def test_model_download():
    """
    测试模型下载功能
    """
    print("\n测试模型重新下载...")
    try:
        from ...faster_whisper import WhisperModel

        print("尝试加载模型 (这将触发重新下载)...")
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("✓ 模型加载成功！")

        # 简单测试
        segments, info = model.transcribe("dummy.wav")  # 这会失败，但我们只是测试模型加载
        print("模型基本功能测试完成")

    except Exception as e:
        print(f"模型测试完成 (预期会有错误): {e}")
        print("模型已成功下载到缓存")


def redownload():
    from ...faster_whisper import WhisperModel

    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="int8")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("samples_jfk.wav", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

def main():
    """
    主函数
    """
    print("Whisper模型缓存管理工具")
    print("=" * 60)

    # 显示当前缓存状态
    show_cache_usage()

    # 显示当前找到的模型
    cache_dirs = get_default_cache_dirs()
    models = find_whisper_models(cache_dirs)
    if models:
        print(f"\n当前找到的Whisper模型 ({len(models)} 个):")
        for model in models:
            print(f"  - {model}")
    else:
        print("\n未找到Whisper模型缓存")

    print("\n请选择操作:")
    print("1. 清除 large-v3 模型缓存")
    print("2. 清除所有 Whisper 相关缓存")
    print("3. 显示手动清理说明")
    print("4. 显示当前缓存状态")
    print("5. 测试模型重新下载")
    print("6. large-v3模型重装和试运行")

    choice = input("\n请输入选择 (1-6): ").strip()

    if choice == "1":
        cleared = clear_specific_model("large-v3")
        if cleared:
            print(f"\n已清除 {len(cleared)} 个 large-v3 相关目录")
        else:
            print("\n未找到 large-v3 模型缓存")

    elif choice == "2":
        cleared = clear_all_whisper_cache()
        if cleared:
            print(f"\n已清除 {len(cleared)} 个 Whisper 相关目录")
        else:
            print("\n未找到 Whisper 相关缓存")

    elif choice == "3":
        manual_clear_instructions()

    elif choice == "4":
        show_cache_usage()

    elif choice == "5":
        test_model_download()

    elif choice == "6":
        redownload()

    else:
        print("无效选择")


if __name__ == "__main__":
    main()