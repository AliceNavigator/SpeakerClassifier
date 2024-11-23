import argparse
import json
import os
import shutil
from modelscope.pipelines import pipeline
from tqdm import tqdm
from colorama import init, Fore, Style
from datetime import datetime
os.environ['MODELSCOPE_CACHE'] = 'pretrained'


def create_output_dir():
    base_output_dir = "./output"
    os.makedirs(base_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")

    spk_lab_dir = os.path.join(run_dir, "target")
    other_dir = os.path.join(run_dir, "other")
    
    for dir_path in [run_dir, spk_lab_dir, other_dir]:
        os.makedirs(dir_path)
        
    return run_dir, spk_lab_dir, other_dir


def main():
    init()
    parser = argparse.ArgumentParser(description='Speaker verification and classification tool')
    parser.add_argument('--target', type=str, required=True, help='Path to target speaker audio file')
    parser.add_argument('--input_dir', type=str, required=True, default='input', help='Directory containing audio files to classify')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for classification (default: 0.6)')
    parser.add_argument('--device', type=str, default='cuda', help='Whether to use cuda or cpu is used to do inference')
    args = parser.parse_args()

    run_dir, spk_lab_dir, other_dir = create_output_dir()

    sv_pipeline = pipeline(
        task='speaker-verification',
        model='damo/speech_eres2net_sv_zh-cn_16k-common',
        model_revision='v1.0.5',
        device=args.device,
    )

    audio_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.wav', '.WAV'))]

    results = {
        "target_speaker": os.path.basename(args.target),
        "threshold": args.threshold,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classifications": []
    }

    files_to_move = {
        "target": [],
        "other": []
    }

    print(f"\nProcessing {len(audio_files)} audio files...")
    print(f"Target speaker: {os.path.basename(args.target)}")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {run_dir}\n")

    pbar = tqdm(audio_files, desc="Verifying speakers")

    for audio_file in pbar:
        audio_path = os.path.join(args.input_dir, audio_file)

        result = sv_pipeline([args.target, audio_path], thr=args.threshold)

        score = result['score']
        is_same_speaker = result['text'] == 'yes'

        status = "TARGET" if is_same_speaker else "OTHER"
        status_color = Fore.GREEN if is_same_speaker else Fore.RED

        pbar.write(
            f"{Fore.CYAN}{audio_file:<30}{Style.RESET_ALL} | "
            f"Score: {Fore.YELLOW}{score:.4f}{Style.RESET_ALL} | "
            f"Status: {status_color}{status}{Style.RESET_ALL}"
        )

        result_entry = {
            "file": audio_file,
            "score": float(score),
            "is_target_speaker": is_same_speaker
        }
        results["classifications"].append(result_entry)

        if is_same_speaker:
            files_to_move["target"].append((audio_path, audio_file))
        else:
            files_to_move["other"].append((audio_path, audio_file))

    results_file = os.path.join(run_dir, "classification_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{Fore.YELLOW}Moving files...{Style.RESET_ALL}")

    for src_path, filename in tqdm(files_to_move["target"], 
                                 desc=f"{Fore.GREEN}Moving target speaker files{Style.RESET_ALL}"):
        shutil.copy2(src_path, os.path.join(spk_lab_dir, filename))

    for src_path, filename in tqdm(files_to_move["other"], 
                                 desc=f"{Fore.RED}Moving other speaker files{Style.RESET_ALL}"):
        shutil.copy2(src_path, os.path.join(other_dir, filename))

    total_files = len(audio_files)
    target_speaker_files = len(files_to_move["target"])
    other_speaker_files = len(files_to_move["other"])
    
    print(f"\n{Fore.GREEN}Classification Complete!{Style.RESET_ALL}")
    print(f"Total files processed: {total_files}")
    print(f"Files classified as target speaker: {Fore.GREEN}{target_speaker_files}{Style.RESET_ALL}")
    print(f"Files classified as other speakers: {Fore.RED}{other_speaker_files}{Style.RESET_ALL}")
    print(f"\nResults saved to: {Fore.CYAN}{results_file}{Style.RESET_ALL}")
    print(f"Classified audio files saved in: {Fore.CYAN}{run_dir}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
