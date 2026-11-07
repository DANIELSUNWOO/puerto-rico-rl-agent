"""[Moka] 학습 체크포인트를 대회 제출 형식의 참가자 폴더로 만듭니다.

체크포인트에서 신경망 숫자(state_dict)만 꺼내 weights.pt로 저장하고 (조교 공지 4번),
my_agent/my_agent.py를 복사하면서 리더보드 이름(name)만 바꿉니다.

사용법:
    python training/export_entry.py <체크포인트.pt> <폴더> <이름>
예:
    python training/export_entry.py results/exp03_mix_lr1e-5/checkpoints/ckpt_final.pt results/entries_exp03/0306_e3_30 E3_30
"""
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(ckpt_path, out_dir, alias):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, "weights.pt"))       # 숫자만 저장

    with open(os.path.join(ROOT, "my_agent", "my_agent.py"), encoding="utf-8") as f:
        src = f.read()
    src, n = re.subn(r'^(\s*name\s*=\s*)"[^"]*"', rf'\1"{alias}"', src, count=1, flags=re.M)
    if n != 1:
        sys.exit("my_agent.py에서 name = \"...\" 줄을 찾지 못했습니다")
    with open(os.path.join(out_dir, "my_agent.py"), "w", encoding="utf-8") as f:
        f.write(src)
    size = os.path.getsize(os.path.join(out_dir, "weights.pt")) / 1e6
    print(f"{out_dir}: name={alias}, weights.pt {size:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    export(*sys.argv[1:])