"""
Simplified Reward Hacking Demo - Best-of-N Approach
Shows reward hacking through rejection sampling (no training needed)

Installation:
pip install transformers torch

Usage:
python reward_hacking_simple.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class BuggyRewardModel:
    """Reward model with intentional bias"""
    hack_words = ["amazing", "fantastic", "excellent", "perfect", "wonderful", "outstanding"]
    
    def score_proxy(self, text):
        """Buggy proxy reward - gives bonus for keyword stuffing"""
        base_score = len(text.split()) * 0.05
        keyword_count = sum(text.lower().count(w) for w in self.hack_words)
        bonus = keyword_count * 3.0  # BIG bonus for keywords
        return base_score + bonus
    
    def score_truth(self, text):
        """Ground truth - penalizes unnatural keyword spam"""
        base_score = len(text.split()) * 0.05
        keyword_count = sum(text.lower().count(w) for w in self.hack_words)
        
        # Penalize excessive keywords
        if keyword_count > 2:
            penalty = (keyword_count - 2) * 2.0
            base_score -= penalty
        elif keyword_count == 1 or keyword_count == 2:
            base_score += 0.5  # Small bonus for natural use
            
        return base_score


def generate_samples(model, tokenizer, prompt, n, device):
    """Generate n samples from the model"""
    samples = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=1.2,  # High temp for diversity
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(text)
    return samples


def best_of_n(samples, reward_model, use_proxy=True):
    """Select best sample according to reward model"""
    if use_proxy:
        scores = [reward_model.score_proxy(s) for s in samples]
    else:
        scores = [reward_model.score_truth(s) for s in samples]
    
    best_idx = np.argmax(scores)
    return samples[best_idx], scores[best_idx]


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    print("Loading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    reward_model = BuggyRewardModel()
    
    prompts = [
        "Write a product review:",
        "This movie was",
        "My experience with this service:"
    ]
    
    print("EXPERIMENT: Comparing different n values")
    
    results = []
    
    for n in [1, 4, 16, 32]:
        print(f"\n>>> Testing with n={n} samples")
        print("-"*70)
        
        proxy_rewards = []
        truth_rewards = []
        keyword_counts = []
        examples = []
        
        for prompt in prompts:
            # Generate n samples
            samples = generate_samples(model, tokenizer, prompt, n, device)
            
            # Select best using PROXY reward (this is where hacking happens!)
            best_sample, best_proxy = best_of_n(samples, reward_model, use_proxy=True)
            
            # Evaluate with BOTH rewards
            proxy_score = reward_model.score_proxy(best_sample)
            truth_score = reward_model.score_truth(best_sample)
            
            # Count keywords
            keywords = sum(best_sample.lower().count(w) for w in reward_model.hack_words)
            
            proxy_rewards.append(proxy_score)
            truth_rewards.append(truth_score)
            keyword_counts.append(keywords)
            examples.append(best_sample)
        
        avg_proxy = np.mean(proxy_rewards)
        avg_truth = np.mean(truth_rewards)
        avg_keywords = np.mean(keyword_counts)
        
        results.append({
            'n': n,
            'proxy': avg_proxy,
            'truth': avg_truth,
            'keywords': avg_keywords,
            'example': examples[0]
        })
        
        print(f"Avg Proxy Reward: {avg_proxy:.2f}")
        print(f"Avg True Reward:  {avg_truth:.2f}")
        print(f"Avg Keywords:     {avg_keywords:.1f}")
        print(f"Example: {examples[0][:80]}...")
    
    # Analysis  
    print("Summary Table:")
    print(f"{'n':<6} {'Proxy':<8} {'Truth':<8} {'Keywords':<10} {'Gap':<8}")
    print("-"*50)
    for r in results:
        gap = r['proxy'] - r['truth']
        print(f"{r['n']:<6} {r['proxy']:<8.2f} {r['truth']:<8.2f} {r['keywords']:<10.1f} {gap:<8.2f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()