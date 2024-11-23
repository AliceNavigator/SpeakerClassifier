import os
import inquirer
import subprocess
from typing import List
from colorama import init, Fore, Style
os.system('title SpeakerClassifier v0.1    by 领航员未鸟')


def get_audio_files(directory: str) -> List[str]:
    audio_extensions = ('.wav', '.WAV')
    audio_files = []

    try:
        for file in os.listdir(directory):
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(directory, file))
    except OSError as e:
        print(f"{Fore.RED}\n访问目录时出错 {directory}: {e}{Fore.RESET}")
        return []

    return audio_files


def run_classifier(target: str, input_dir: str, threshold: float, device: str):
    try:
        cmd = [
            "python", "speaker_classifier.py",
            "--target", target,
            "--input_dir", input_dir,
            "--threshold", str(threshold),
            "--device", device
        ]

        print(f"\n开始对符合特征的音频进行分类: {Fore.CYAN}{os.path.basename(target)}{Fore.RESET}\n")

        process = subprocess.run(cmd, check=True, text=True)
        print(f"\n分类已完成: {Fore.CYAN}{os.path.basename(target)}{Fore.RESET}\n")

    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}运行分类时出错: {e}{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}未知错误： {e}{Fore.RESET}")


def main():
    spk_labels_dir = "spk_labels"
    if not os.path.exists(spk_labels_dir):
        print(f"{Fore.RED}错误: {spk_labels_dir} 目录未找到！{Fore.RESET}")
        return

    target_files = get_audio_files(spk_labels_dir)
    if not target_files:
        print(f"{Fore.RED}{spk_labels_dir} 中未找到可用音频！{Fore.RESET}")
        return

    print(f"\n==========================================================================================")
    print(f"● 本工具旨在通过声纹快速将音频数据集中特定说话人单独说话的音频片段分类出来，用来进行数据清洗")
    print(f"   使用阿里开源模型：https://www.modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common/\n")
    print(f" ○ 请保证所有的音频都是wav格式，提取完毕人声并已完成切片的状态，不会的话可以参照以下教程\n")
    print(f" ○ 提取人声：BV1t6sUeTEwa  转格式，切片：BV1qE28YKEHm  或者参照我动态置顶\n")
    print(f" ○ 对于所需的目标说话人音色参考音频只需10秒左右即可，但其质量将很大程度影响分类质量\n")
    print(f" ○ 可以对歌声数据识别并分类，排除含和声片段，准确度不如说话，可考虑适当放宽或收紧阈值")
    print(f"==========================================================================================\n")

    print(f"{Fore.YELLOW}  ◆ 使用本工具并不意味着大杂烩语音切好丢进去就能把你要的人提出来，\n  ◇ 你只会收获少的可怜的数据或者主要是目标说话人但也有其他人说话的片段。{Fore.RESET}")
    print(f"{Fore.YELLOW}\n  ◆ 最方便有效的办法依然是选择素材时就做出挑选，并将本工具视作排除意外干扰的手段{Fore.RESET}")
    print(f"{Fore.YELLOW}  ◇ （如数个小时的原始素材中可能在很散的地方有人零星说了几句话或者放了几首含人声的歌的情况）\n{Fore.RESET}")

    questions = [
        inquirer.List('target',
                      message=f"{Fore.BLACK}{Style.BRIGHT}请选择可代表目标说话人音色特征的音频片段，应放入 {spk_labels_dir} 中{Style.RESET_ALL}",
                      choices=[(os.path.basename(f), f) for f in target_files]),
        inquirer.List('device',
                      message=f"{Fore.BLACK}{Style.BRIGHT}选择推理所用设备{Style.RESET_ALL}",
                      choices=['cpu', 'cuda'],
                      default='cuda'),
        inquirer.Text('threshold',
                      message=f"{Fore.BLACK}{Style.BRIGHT}请输入判定阈值，置信度大于该值则认为特征相符{Style.RESET_ALL}",
                      default='0.6',
                      validate=lambda _, x: x.replace('.', '').isdigit()),
        inquirer.Text('input_dir',
                      message=f"{Fore.BLACK}{Style.BRIGHT}输入待分离音频所在目录{Style.RESET_ALL}",
                      default='input',
                      validate=lambda _, x: os.path.exists(x))
    ]

    try:
        answers = inquirer.prompt(questions)
        if not answers:
            print(f"{Fore.YELLOW}\n操作已被用户强制取消\n{Fore.RESET}")
            return

        threshold = float(answers['threshold'])
        if not 0 <= threshold <= 1:
            print(f"{Fore.YELLOW}警告： 阈值只能在0-1之间！{Fore.RESET}")
            return

        run_classifier(
            target=answers['target'],
            input_dir=answers['input_dir'],
            threshold=threshold,
            device=answers['device']
        )

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\n操作已被用户强制取消\n{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}错误: {e}{Fore.RESET}")


if __name__ == "__main__":
    init()
    main()
