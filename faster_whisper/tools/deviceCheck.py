import platform
import sys


def check_system_info():
    """显示系统基本信息"""
    print("=" * 50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print("=" * 50)


def check_ct2_supported_compute_types():
    """检查CTranslate2支持的计算类型"""
    try:
        import ctranslate2
        print("CTranslate2信息:")
        print(f"CTranslate2版本: {ctranslate2.__version__}")

        # 检查支持的计算类型
        supported_compute_types = ctranslate2.get_supported_compute_types("cpu")
        print(f"CPU支持的计算类型: {supported_compute_types}")

        # 检查GPU支持（如果有）
        devices = ctranslate2.get_cuda_device_count()
        if devices > 0:
            print(f"CUDA设备数量: {devices}")
            supported_compute_types_gpu = ctranslate2.get_supported_compute_types("cuda")
            print(f"CUDA支持的计算类型: {supported_compute_types_gpu}")

            # 显示GPU详细信息
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(devices):
                        gpu_name = torch.cuda.get_device_name(i)
                        print(f"GPU {i}: {gpu_name}")

                        # 检查GPU内存
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        print(f"  总内存: {total_memory // (1024 ** 2)} MB")
            except ImportError:
                print("注意: PyTorch未安装，无法获取GPU详细信息")
        else:
            print("未检测到CUDA设备")

    except ImportError:
        print("错误: CTranslate2未安装")
        print("请运行: pip install ctranslate2")
        return False
    except Exception as e:
        print(f"检查CTranslate2支持时出错: {e}")
        return False

    return True


def check_torch_cuda_info():
    """检查PyTorch和CUDA信息"""
    try:
        import torch
        print("\nPyTorch信息:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cudnn版本: {torch.backends.cudnn.version()}")
            print(f"cudnn可用: {torch.backends.cudnn.enabled}")

            # 检查Tensor Core支持
            if hasattr(torch.cuda, 'is_bf16_supported'):
                print(f"BF16支持: {torch.cuda.is_bf16_supported()}")
            print(
                f"半精度浮点支持: {torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else 'N/A'}")

        return True
    except ImportError:
        print("\nPyTorch未安装，跳过CUDA检查")
        return False
    except Exception as e:
        print(f"检查PyTorch信息时出错: {e}")
        return False


def check_faster_whisper_support():
    """检查Faster Whisper支持的计算类型"""
    try:
        from ...faster_whisper.utils import get_assets_audio
        import ctranslate2
        print("\nFaster Whisper支持信息:")

        # 显示所有可能的计算类型
        all_possible_types = ["int8", "int8_float16", "int16", "float16", "float32"]
        print("所有可能的计算类型:", all_possible_types)

        # 检查CPU支持的类型
        cpu_supported = ctranslate2.get_supported_compute_types("cpu")
        print(f"CPU实际支持: {cpu_supported}")

        # 检查GPU支持的类型（如果可用）
        if ctranslate2.get_cuda_device_count() > 0:
            gpu_supported = ctranslate2.get_supported_compute_types("cuda")
            print(f"GPU实际支持: {gpu_supported}")

        return True
    except ImportError:
        print("\nFaster Whisper未安装，跳过相关检查")
        return False
    except Exception as e:
        print(f"检查Faster Whisper支持时出错: {e}")
        return False


def suggest_optimal_settings():
    """建议最优设置"""
    try:
        import ctranslate2
        import torch

        print("\n建议的最优设置:")

        if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
            gpu_supported = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in gpu_supported:
                print("- 推荐使用: device='cuda', compute_type='float16' (最快且节省显存)")
            elif "float32" in gpu_supported:
                print("- 推荐使用: device='cuda', compute_type='float32' (CUDA可用但不支持float16)")
            else:
                print("- 推荐使用: device='cuda', compute_type='int8' (如果支持)")
        else:
            cpu_supported = ctranslate2.get_supported_compute_types("cpu")
            if "float32" in cpu_supported:
                print("- 推荐使用: device='cpu', compute_type='float32' (CPU模式)")
            elif "int8" in cpu_supported:
                print("- 推荐使用: device='cpu', compute_type='int8' (CPU模式)")
            else:
                print("- 推荐使用: device='cpu', compute_type='int16' (CPU模式)")

        # 内存建议
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 6 * 1024 ** 3:  # 6GB
                print("- GPU显存较低，建议使用较小的模型或CPU模式")
            elif gpu_memory < 12 * 1024 ** 3:  # 12GB
                print("- GPU显存适中，可使用medium或large模型")
            else:
                print("- GPU显存充足，可使用最大模型")

    except Exception as e:
        print(f"\n生成建议时出错: {e}")


def main():
    """主函数"""
    print("本地系统计算类型支持检查工具")
    print("=" * 60)

    check_system_info()
    success1 = check_ct2_supported_compute_types()
    success2 = check_torch_cuda_info()
    success3 = check_faster_whisper_support()

    if success1:
        suggest_optimal_settings()

    print("\n" + "=" * 60)
    print("检查完成!")


if __name__ == "__main__":
    main()
