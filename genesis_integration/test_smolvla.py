import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    print("🚀 Loading smolVLA...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", device="cuda")
    policy.eval()

    print("📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    state_dim = policy.config.input_features["observation.state"].shape
    print(f"   state_dim = {state_dim}")

    dummy_pixels = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device="cuda")
    dummy_state  = torch.zeros((1, *state_dim),  dtype=torch.float32, device="cuda")

    instruction = ["pick up the red block and place it on the green mat"]
    enc = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    observation = {
        "observation.images.camera1":          dummy_pixels,
        "observation.state":                   dummy_state,
        "observation.language.tokens":         enc["input_ids"].to("cuda"),
        "observation.language.attention_mask": enc["attention_mask"].bool().to("cuda"),  # ← fix
    }

    print("🧠 Running inference test...")
    try:
        with torch.no_grad():
            action_chunk = policy.select_action(observation)
        print(f"\n✅ TEST PASSED — action shape: {action_chunk.shape}")
        print(f"   First step:\n{action_chunk.cpu().numpy()}")
    except Exception as e:
        import traceback
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()