"""
Diagnostic: check whether Qwen2.5-VL-3B-Instruct can see a Sokoban image.
Usage:
    python diagnose_vision.py --image ./test_seed0/step_0.png
    python diagnose_vision.py --image ./test_seed0/step_0.png --model_path /path/to/local/model
"""
import argparse
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to Sokoban PNG image")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID or local path",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded.\n")

    img = Image.open(args.image).convert("RGB")
    print(f"Image size: {img.size}\n")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": (
                        "This is a screenshot from a Sokoban puzzle game.\n"
                        "Please describe what you see: what colors and shapes are present, "
                        "and where is each element located (e.g., top-left, center, bottom-right)?"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    # Strip prompt tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    print("=" * 60)
    print("Model response:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
